import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

# =========================
# Parameters (edit these)
# =========================
RADIUS_M = 1000.0
TP_DBM = 5.0
TPACKET_S = 0.050
NCHANNELS = 8
CAPTURE_DB = 3.0

TOTAL_PACKETS = 10_000          # fixed by project description
MEAN_IAT_PER_DEVICE = 600.0     # seconds

# Experiment sweep (edit as you like)
N_VALUES = [500, 1000, 2000, 5000, 10_000]

# Clustering options (fixed by description)
C_OPTIONS = [1, 2, 4]

# SA/local search tuning
RESTARTS = 6
ITERS_PER_RESTART = 4000
SA_START_TEMP = 0.002
SA_END_TEMP = 0.0

SEED = 12345

# =========================
# Debug printing controls
# =========================
PRINT_SA_PROGRESS = False
SA_PRINT_STEPS = [0, 50, 200, 500, 1000, 2000, 3000, 3999]  # keep last = ITERS_PER_RESTART-1

# =========================
# CDF plot controls
# =========================
CDF_PLOT_N = max(N_VALUES)     # which N to use for the CDF plot
CDF_SCENARIOS = ["baseline", "C=1", "C=2", "C=4"]  # which scenarios to include in CDF


# =========================
# Helpers: geometry + RX
# =========================
def sample_devices_uniform_disk(n: int, radius_m: float, rng: np.random.Generator):
    """Uniform in area inside disk."""
    u = rng.random(n)
    r = np.sqrt(u) * radius_m
    theta = rng.random(n) * 2.0 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    d = np.sqrt(x * x + y * y)
    return x, y, d

def path_loss_db(d_m: np.ndarray):
    """PL(d)=40+27*log10(d_meters). Avoid log10(0)."""
    d_safe = np.maximum(d_m, 1.0)
    return 40.0 + 27.0 * np.log10(d_safe)

def compute_rx_dbm(d_m: np.ndarray):
    """RX = TP - PL(d)."""
    return TP_DBM - path_loss_db(d_m)


# =========================
# Traffic schedule
# =========================
@dataclass
class Schedule:
    t_start: np.ndarray
    dev_id: np.ndarray
    u_chan: np.ndarray

def generate_schedule(n_devices: int, total_packets: int, rng: np.random.Generator) -> Schedule:
    mean_iat = MEAN_IAT_PER_DEVICE / float(n_devices)
    iats = rng.exponential(scale=mean_iat, size=total_packets)
    t_start = np.cumsum(iats)
    dev_id = rng.integers(0, n_devices, size=total_packets, endpoint=False)
    u_chan = rng.random(total_packets)
    return Schedule(t_start=t_start, dev_id=dev_id, u_chan=u_chan)


# =========================
# Channel assignment per packet
# =========================
def baseline_channels(u_chan: np.ndarray) -> np.ndarray:
    return np.floor(u_chan * NCHANNELS).astype(np.int32)

def clustered_channels(dev_id: np.ndarray, u_chan: np.ndarray, dev_cluster: np.ndarray, C: int) -> np.ndarray:
    k = dev_cluster[dev_id]
    within = np.floor(u_chan * C).astype(np.int32)
    return (k * C + within).astype(np.int32)


# =========================
# Core simulation (collision + capture)
# =========================
@dataclass
class SimResult:
    success_count: int
    collision_fail_count: int
    collided_packet_count: int
    capture_success_count: int
    success_prob: float
    per_packet_success: np.ndarray

def simulate(schedule: Schedule, rx_dbm: np.ndarray, channels: np.ndarray) -> SimResult:
    P = schedule.t_start.shape[0]
    t0 = schedule.t_start
    t1 = t0 + TPACKET_S
    dev = schedule.dev_id
    rx = rx_dbm[dev]

    per_packet_success = np.zeros(P, dtype=bool)

    collided_packet_count = 0
    capture_success_count = 0

    for ch in range(NCHANNELS):
        idx = np.where(channels == ch)[0]
        if idx.size == 0:
            continue

        order = np.argsort(t0[idx])
        idx = idx[order]

        comp = [idx[0]]
        comp_end = t1[idx[0]]

        def resolve_component(comp_list):
            nonlocal collided_packet_count, capture_success_count
            if len(comp_list) == 1:
                per_packet_success[comp_list[0]] = True
                return

            collided_packet_count += len(comp_list)

            comp_rx = rx[np.array(comp_list)]
            best_pos = int(np.argmax(comp_rx))
            best_idx = comp_list[best_pos]
            best_rx = comp_rx[best_pos]

            ok = True
            for pkt in comp_list:
                if pkt == best_idx:
                    continue
                if best_rx - rx[pkt] < CAPTURE_DB:
                    ok = False
                    break

            if ok:
                per_packet_success[best_idx] = True
                capture_success_count += 1

        for pkt in idx[1:]:
            if t0[pkt] < comp_end:
                comp.append(pkt)
                if t1[pkt] > comp_end:
                    comp_end = t1[pkt]
            else:
                resolve_component(comp)
                comp = [pkt]
                comp_end = t1[pkt]

        resolve_component(comp)

    success_count = int(per_packet_success.sum())
    collision_fail_count = int(P - success_count)
    success_prob = success_count / float(P)

    return SimResult(
        success_count=success_count,
        collision_fail_count=collision_fail_count,
        collided_packet_count=int(collided_packet_count),
        capture_success_count=int(capture_success_count),
        success_prob=success_prob,
        per_packet_success=per_packet_success
    )


# =========================
# Optimizer: stratified init + SA swaps + restarts
# =========================
def stratified_interleaved_order(rx_dbm: np.ndarray) -> np.ndarray:
    n = rx_dbm.size
    order = np.argsort(rx_dbm)
    lo, hi = 0, n - 1
    inter = np.empty(n, dtype=np.int32)
    k = 0
    while lo <= hi:
        inter[k] = order[hi]; k += 1; hi -= 1
        if lo <= hi:
            inter[k] = order[lo]; k += 1; lo += 1
    return inter

def initial_stratified_assignment(rx_dbm: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n = rx_dbm.size
    inter = stratified_interleaved_order(rx_dbm)
    shift = int(rng.integers(0, num_clusters))
    dev_cluster = np.empty(n, dtype=np.int32)
    for i, dev in enumerate(inter):
        dev_cluster[dev] = (i + shift) % num_clusters
    return dev_cluster

def local_swap_improve_sa(dev_cluster: np.ndarray,
                          schedule: Schedule,
                          rx_dbm: np.ndarray,
                          C: int,
                          rng: np.random.Generator,
                          iters: int = ITERS_PER_RESTART,
                          start_temp: float = SA_START_TEMP,
                          end_temp: float = SA_END_TEMP,
                          restart_id: int = -1,
                          N_devices: int = -1) -> np.ndarray:
    n = rx_dbm.size

    def fitness(dc: np.ndarray) -> float:
        ch = clustered_channels(schedule.dev_id, schedule.u_chan, dc, C)
        return simulate(schedule, rx_dbm, ch).success_prob

    best = dev_cluster.copy()
    best_fit = fitness(best)

    cur = best.copy()
    cur_fit = best_fit

    if PRINT_SA_PROGRESS:
        print(f"\n  [SA start] N={N_devices}, C={C}, restart={restart_id}, iters={iters}, init_fit={best_fit:.6f}")

    for t in range(iters):
        if iters > 1:
            temp = start_temp + (end_temp - start_temp) * (t / (iters - 1))
        else:
            temp = end_temp

        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if cur[a] == cur[b]:
            continue

        ca, cb = cur[a], cur[b]
        cur[a], cur[b] = cb, ca

        new_fit = fitness(cur)
        delta = new_fit - cur_fit

        accept = False
        reason = "reject"
        if delta >= 0:
            accept = True
            reason = "improve"
        elif temp > 0:
            p = np.exp(delta / temp)
            if rng.random() < p:
                accept = True
                reason = "worse-accept"

        if accept:
            cur_fit = new_fit
            if new_fit > best_fit:
                best_fit = new_fit
                best = cur.copy()
        else:
            cur[a], cur[b] = ca, cb

        if PRINT_SA_PROGRESS:
            if (t in SA_PRINT_STEPS) or (t == iters - 1):
                print(f"    iter={t:4d}  temp={temp:.6f}  "
                      f"cur_fit={cur_fit:.6f}  best_fit={best_fit:.6f}  "
                      f"delta={delta:+.6f}  {reason}")

    if PRINT_SA_PROGRESS:
        print(f"  [SA end]   N={N_devices}, C={C}, restart={restart_id}, best_fit={best_fit:.6f}\n")

    return best

def optimize_assignment(rx_dbm: np.ndarray,
                        schedule: Schedule,
                        C: int,
                        rng: np.random.Generator,
                        restarts: int = RESTARTS,
                        iters_per_restart: int = ITERS_PER_RESTART,
                        N_devices: int = -1) -> np.ndarray:
    num_clusters = NCHANNELS // C

    best_dc = None
    best_fit = -1.0

    def fitness(dc: np.ndarray) -> float:
        ch = clustered_channels(schedule.dev_id, schedule.u_chan, dc, C)
        return simulate(schedule, rx_dbm, ch).success_prob

    if PRINT_SA_PROGRESS:
        print(f"== Optimizing assignment for N={N_devices}, C={C} "
              f"(clusters={num_clusters}, restarts={restarts}, iters={iters_per_restart}) ==")

    for r in range(restarts):
        dc0 = initial_stratified_assignment(rx_dbm, num_clusters, rng)
        init_fit = fitness(dc0)
        if PRINT_SA_PROGRESS:
            print(f"  [restart {r+1}/{restarts}] init_fit={init_fit:.6f}")

        dc = local_swap_improve_sa(dc0, schedule, rx_dbm, C, rng,
                                   iters=iters_per_restart,
                                   start_temp=SA_START_TEMP,
                                   end_temp=SA_END_TEMP,
                                   restart_id=(r+1),
                                   N_devices=N_devices)

        f = fitness(dc)
        if PRINT_SA_PROGRESS:
            print(f"  [restart {r+1}/{restarts}] final_fit={f:.6f}")

        if f > best_fit:
            best_fit = f
            best_dc = dc.copy()

    if PRINT_SA_PROGRESS:
        print(f"== Best over restarts: N={N_devices}, C={C}, best_fit={best_fit:.6f} ==\n")

    return best_dc


# =========================
# NEW: CDF plot (per-device success prob vs distance)
# =========================
def compute_device_success_probs(n_devices: int, schedule: Schedule, per_packet_success: np.ndarray):
    """
    Returns:
      tx[i] = number of transmissions device i attempted
      p[i]  = success probability for device i (only defined where tx>0)
    """
    tx = np.zeros(n_devices, dtype=np.int32)
    sx = np.zeros(n_devices, dtype=np.int32)

    dev = schedule.dev_id
    np.add.at(tx, dev, 1)
    np.add.at(sx, dev, per_packet_success.astype(np.int32))

    p = np.zeros(n_devices, dtype=float)
    mask = tx > 0
    p[mask] = sx[mask] / tx[mask]
    return tx, p, mask

def plot_cdf_success_vs_distance(distance_m: np.ndarray,
                                 schedule: Schedule,
                                 per_packet_success: np.ndarray,
                                 title: str):
    """
    "CDF-style" curve requested in description:
      - x-axis: distance
      - y-axis: cumulative fraction of devices (sorted by distance) with their per-device success probability.
    We plot the running average success probability vs distance (smooth), and also allow a true CDF of success probs.
    Here we do a clean CDF of per-device success probability (y) against distance (x) by sorting devices by distance.
    """
    n = distance_m.size
    tx, p, mask = compute_device_success_probs(n, schedule, per_packet_success)

    # Only devices that transmitted at least once in the 10k packets
    d = distance_m[mask]
    p = p[mask]

    order = np.argsort(d)
    d = d[order]
    p = p[order]

    # Running mean of per-device success probability vs distance (smooth "CDF-like" curve)
    running_mean = np.cumsum(p) / (np.arange(p.size) + 1)

    plt.figure()
    plt.plot(d, running_mean, label="Running mean success prob")
    plt.xlabel("Distance to gateway (m)")
    plt.ylabel("Success probability")
    plt.title(title)
    plt.grid(True)
    plt.legend()


# =========================
# Main experiment
# =========================
def run():
    rng_master = np.random.default_rng(SEED)

    scenarios = ["baseline"] + [f"C={C}" for C in C_OPTIONS]
    success_counts = {s: [] for s in scenarios}
    collision_counts = {s: [] for s in scenarios}
    success_probs = {s: [] for s in scenarios}
    runtimes = {s: [] for s in scenarios}

    # store per-packet success for CDF plot for N=CDF_PLOT_N
    cdf_store = {}  # scenario -> (distance, schedule, per_packet_success)
    cdf_distance = None
    cdf_schedule = None

    for N in N_VALUES:
        print(f"\n=== N = {N} devices ===")
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        # Devices + RX
        _, _, d = sample_devices_uniform_disk(N, RADIUS_M, rng)
        rx_dbm = compute_rx_dbm(d)

        # Schedule (fixed per N)
        schedule = generate_schedule(N, TOTAL_PACKETS, rng)

        # ----- Baseline -----
        t0 = time.perf_counter()
        ch_base = baseline_channels(schedule.u_chan)
        res_base = simulate(schedule, rx_dbm, ch_base)
        t1 = time.perf_counter()

        success_counts["baseline"].append(res_base.success_count)
        collision_counts["baseline"].append(res_base.collision_fail_count)
        success_probs["baseline"].append(res_base.success_prob)
        runtimes["baseline"].append(t1 - t0)

        print(f"baseline: success_prob={res_base.success_prob:.4f}, "
              f"success={res_base.success_count}, collisions={res_base.collision_fail_count}, "
              f"time={t1 - t0:.3f}s")

        if N == CDF_PLOT_N and "baseline" in CDF_SCENARIOS:
            cdf_store["baseline"] = (d.copy(), schedule, res_base.per_packet_success.copy())

        # ----- Clustering options -----
        for C in C_OPTIONS:
            label = f"C={C}"

            t0 = time.perf_counter()

            dc = optimize_assignment(rx_dbm, schedule, C, rng,
                                     restarts=RESTARTS,
                                     iters_per_restart=ITERS_PER_RESTART,
                                     N_devices=N)

            ch = clustered_channels(schedule.dev_id, schedule.u_chan, dc, C)
            res = simulate(schedule, rx_dbm, ch)

            t1 = time.perf_counter()

            success_counts[label].append(res.success_count)
            collision_counts[label].append(res.collision_fail_count)
            success_probs[label].append(res.success_prob)
            runtimes[label].append(t1 - t0)

            print(f"{label}: success_prob={res.success_prob:.4f}, "
                  f"success={res.success_count}, collisions={res.collision_fail_count}, "
                  f"time={t1 - t0:.3f}s")

            if N == CDF_PLOT_N and label in CDF_SCENARIOS:
                cdf_store[label] = (d.copy(), schedule, res.per_packet_success.copy())

    # =========================
    # Plots required by project
    # =========================
    Ns = np.array(N_VALUES)

    # 1) Success count vs N
    plt.figure()
    for s in scenarios:
        plt.plot(Ns, success_counts[s], marker='o', label=s)
    plt.xlabel("Number of devices (N)")
    plt.ylabel("Successful transmissions (count)")
    plt.title("Successful transmissions vs N")
    plt.grid(True)
    plt.legend()

    # 2) Collisions vs N (failures)
    plt.figure()
    for s in scenarios:
        plt.plot(Ns, collision_counts[s], marker='o', label=s)
    plt.xlabel("Number of devices (N)")
    plt.ylabel("Collisions / failed transmissions (count)")
    plt.title("Collisions vs N")
    plt.grid(True)
    plt.legend()

    # 3) Success probability vs N
    plt.figure()
    for s in scenarios:
        plt.plot(Ns, success_probs[s], marker='o', label=s)
    plt.xlabel("Number of devices (N)")
    plt.ylabel("Success probability")
    plt.title("Success probability vs N")
    plt.grid(True)
    plt.legend()

    # 4) Runtime vs N
    plt.figure()
    for s in scenarios:
        plt.plot(Ns, runtimes[s], marker='o', label=s)
    plt.xlabel("Number of devices (N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs N")
    plt.grid(True)
    plt.legend()

    # 5) NEW: CDF-style plot(s) vs distance for selected scenarios at N=CDF_PLOT_N
    if len(cdf_store) > 0:
        for label, (dist, sched, per_succ) in cdf_store.items():
            plot_cdf_success_vs_distance(dist, sched, per_succ,
                                         title=f"CDF-style success vs distance (N={CDF_PLOT_N}, {label})")

    plt.show()


if __name__ == "__main__":
    run()
