from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Global constants (from spec)
# -------------------------------
RADIUS_KM = 1.0                 # disk radius
TP_DBM = 5.0                    # transmit power for all devices
PACKET_DURATION = 0.05          # 50 ms
NUM_CHANNELS = 8
CAPTURE_THRESHOLD_DB = 3.0
MEAN_INTERARRIVAL_PER_DEVICE = 600.0  # seconds
TOTAL_PACKETS = 10000


# -------------------------------
# Topology
# -------------------------------
@dataclass
class Device:
    id: int
    x: float
    y: float
    distance_km: float
    rx_power_dbm: float


def generate_devices(N: int, seed: Optional[int] = None) -> List[Device]:
    """
    Generate N devices uniformly over disk (area-uniform), compute RX power from path loss model.
    """
    rng = random.Random(seed)

    devices: List[Device] = []
    for i in range(N):
        # Uniform in area: r = R * sqrt(U), theta = 2πV
        u = rng.random()
        v = rng.random()
        r_km = RADIUS_KM * math.sqrt(u)
        theta = 2 * math.pi * v

        x = r_km * math.cos(theta)
        y = r_km * math.sin(theta)

        # Path-loss expects meters; guard against log10(0) if u=0 exactly.
        d_m = max(r_km * 1000.0, 1e-6)
        pl_db = 40.0 + 27.0 * math.log10(d_m)
        rx_dbm = TP_DBM - pl_db

        devices.append(Device(
            id=i,
            x=x,
            y=y,
            distance_km=r_km,
            rx_power_dbm=rx_dbm
        ))
    return devices


# -------------------------------
# Clustering helpers
# -------------------------------
def build_cluster_channel_map(cluster_size: int) -> Dict[int, List[int]]:
    """
    Split the 8 channels into disjoint cluster groups of size C.
      C=1 -> 8 clusters * 1 channel
      C=2 -> 4 clusters * 2 channels
      C=4 -> 2 clusters * 4 channels
    """
    if NUM_CHANNELS % cluster_size != 0:
        raise ValueError("cluster_size must divide NUM_CHANNELS")

    num_clusters = NUM_CHANNELS // cluster_size
    mapping: Dict[int, List[int]] = {}

    ch = 0
    for c in range(num_clusters):
        mapping[c] = list(range(ch, ch + cluster_size))
        ch += cluster_size
    return mapping


def assignment_random(N: int, num_clusters: int, seed: Optional[int] = None) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(num_clusters) for _ in range(N)]


def assignment_distance_round_robin(devices: List[Device], num_clusters: int) -> List[int]:
    """
    Sort by distance then assign round-robin across clusters.
    Returns assignment list where assignment[device_id] = cluster_id.
    """
    order = sorted(devices, key=lambda d: d.distance_km)
    assignment = [-1] * len(devices)
    for idx, dev in enumerate(order):
        assignment[dev.id] = idx % num_clusters
    return assignment


# -------------------------------
# Simulation core
# -------------------------------
def _evaluate_collisions_and_success(packets: List[Dict[str, Any]]) -> None:
    """
    For each channel, group packets into transitive-overlap groups.
    Apply capture rule (3 dB):
      - If one packet in group: success.
      - Else: strongest succeeds iff it is UNIQUE strongest and ≥3 dB above every other packet.
    """
    for p in packets:
        p["success"] = False
        p["collided"] = False

    by_channel: Dict[int, List[int]] = {}
    for idx, p in enumerate(packets):
        by_channel.setdefault(p["channel"], []).append(idx)

    for _ch, idx_list in by_channel.items():
        idx_list_sorted = sorted(idx_list, key=lambda i: packets[i]["start_time"])
        if not idx_list_sorted:
            continue

        current_group = [idx_list_sorted[0]]
        group_end = packets[idx_list_sorted[0]]["end_time"]

        def process_group(group: List[int]) -> None:
            if not group:
                return
            if len(group) == 1:
                i = group[0]
                packets[i]["success"] = True
                packets[i]["collided"] = False
                return

            rx = np.array([packets[i]["rx"] for i in group], dtype=float)
            max_rx = float(rx.max())
            tol = 1e-12
            max_idxs = [i for i in group if abs(packets[i]["rx"] - max_rx) <= tol]
            if len(max_idxs) != 1:
                for i in group:
                    packets[i]["success"] = False
                    packets[i]["collided"] = True
                return

            winner = max_idxs[0]
            capture_ok = True
            for i in group:
                if i == winner:
                    continue
                if max_rx - packets[i]["rx"] < CAPTURE_THRESHOLD_DB:
                    capture_ok = False
                    break

            if capture_ok:
                for i in group:
                    if i == winner:
                        packets[i]["success"] = True
                        packets[i]["collided"] = False
                    else:
                        packets[i]["success"] = False
                        packets[i]["collided"] = True
            else:
                for i in group:
                    packets[i]["success"] = False
                    packets[i]["collided"] = True

        for idx in idx_list_sorted[1:]:
            p = packets[idx]
            # overlap if start < group_end (start==end => no overlap)
            if p["start_time"] < group_end:
                current_group.append(idx)
                group_end = max(group_end, p["end_time"])
            else:
                process_group(current_group)
                current_group = [idx]
                group_end = p["end_time"]

        process_group(current_group)


def run_simulation_with_devices(
    devices: List[Device],
    mode: str,
    cluster_size: int,
    assignment: Optional[List[int]] = None,
    total_packets: int = TOTAL_PACKETS,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run baseline or clustered simulation on a FIXED topology (devices list).

    - baseline: each packet picks random channel 0..7.
    - clustered: device uses its assigned cluster; picks random among C channels in that cluster.
      `assignment` must be a list length N with cluster ids in [0, num_clusters-1].
    """
    if mode not in ("baseline", "clustered"):
        raise ValueError("mode must be 'baseline' or 'clustered'")

    if NUM_CHANNELS % cluster_size != 0:
        raise ValueError("cluster_size must divide NUM_CHANNELS")

    N = len(devices)

    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    cluster_channels: Optional[Dict[int, List[int]]] = None
    num_clusters = NUM_CHANNELS // cluster_size

    if mode == "clustered":
        if assignment is None:
            assignment = assignment_random(N, num_clusters, seed=seed)
        if len(assignment) != N:
            raise ValueError("assignment length must equal N")
        for cid in assignment:
            if cid < 0 or cid >= num_clusters:
                raise ValueError("cluster id out of range in assignment")
        cluster_channels = build_cluster_channel_map(cluster_size)

    # Traffic (aggregate Poisson)
    delta_mean = MEAN_INTERARRIVAL_PER_DEVICE / N
    time_now = 0.0

    packets: List[Dict[str, Any]] = []
    for _ in range(total_packets):
        time_now += float(rng_np.exponential(delta_mean))

        dev = devices[rng_py.randrange(N)]
        if mode == "baseline":
            channel = rng_py.randrange(NUM_CHANNELS)
        else:
            assert assignment is not None and cluster_channels is not None
            ch_list = cluster_channels[assignment[dev.id]]
            channel = ch_list[rng_py.randrange(len(ch_list))]

        packets.append({
            "device_id": dev.id,
            "channel": channel,
            "start_time": time_now,
            "end_time": time_now + PACKET_DURATION,
            "rx": dev.rx_power_dbm,
        })

    _evaluate_collisions_and_success(packets)

    sent = np.zeros(N, dtype=int)
    succ = np.zeros(N, dtype=int)
    total_success = 0
    total_collided = 0

    for p in packets:
        did = p["device_id"]
        sent[did] += 1
        if p["success"]:
            succ[did] += 1
            total_success += 1
        if p["collided"]:
            total_collided += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        per_device_success = np.where(sent > 0, succ / sent, 0.0)

    distances_km = np.array([d.distance_km for d in devices], dtype=float)

    return {
        "mode": mode,
        "N": N,
        "cluster_size": cluster_size,
        "num_clusters": num_clusters,
        "global_success": total_success / total_packets,
        "collision_rate": total_collided / total_packets,
        "per_device_distance_km": distances_km,
        "per_device_success": per_device_success,
        "packets": packets,
    }


# -------------------------------
# GA utilities
# -------------------------------
def _repair_balance(indiv: List[int], num_clusters: int, rng: random.Random) -> None:
    """
    Push the assignment toward balanced cluster sizes.
    Helps because unbalanced clusters overload some channel groups -> more collisions.
    """
    N = len(indiv)
    target = N // num_clusters

    counts = [0] * num_clusters
    for cid in indiv:
        counts[cid] += 1

    over = [c for c in range(num_clusters) if counts[c] > target + 1]
    under = [c for c in range(num_clusters) if counts[c] < target - 1]
    if not over or not under:
        return

    by_cluster: List[List[int]] = [[] for _ in range(num_clusters)]
    for i, cid in enumerate(indiv):
        by_cluster[cid].append(i)

    while over and under:
        c_over = rng.choice(over)
        c_under = rng.choice(under)

        idx = rng.choice(by_cluster[c_over])
        indiv[idx] = c_under

        by_cluster[c_over].remove(idx)
        by_cluster[c_under].append(idx)
        counts[c_over] -= 1
        counts[c_under] += 1

        if counts[c_over] <= target + 1:
            over.remove(c_over)
        if counts[c_under] >= target - 1:
            under.remove(c_under)


def _tournament_selection(pop: List[List[int]], fits: List[float], k: int, rng: random.Random) -> List[int]:
    best_i = None
    best_f = -1e18
    for _ in range(k):
        i = rng.randrange(len(pop))
        if best_i is None or fits[i] > best_f:
            best_i = i
            best_f = fits[i]
    return pop[best_i][:]


def _uniform_crossover(p1: List[int], p2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    """
    Uniform crossover works better than one-point crossover because device index order
    has no structural meaning.
    """
    if len(p1) != len(p2):
        raise ValueError("parents must have same length")
    c1 = p1[:]
    c2 = p2[:]
    for i in range(len(p1)):
        if rng.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def _mutate(indiv: List[int], num_clusters: int, per_gene_rate: float, rng: random.Random) -> None:
    for i in range(len(indiv)):
        if rng.random() < per_gene_rate:
            indiv[i] = rng.randrange(num_clusters)


def _evaluate_assignment(
    devices: List[Device],
    assignment: List[int],
    cluster_size: int,
    packets_for_fitness: int,
    traffic_seeds: List[int],
) -> float:
    """
    Fitness = average global success over few traffic seeds (reduces noise).
    """
    scores = []
    for s in traffic_seeds:
        m = run_simulation_with_devices(
            devices=devices,
            mode="clustered",
            cluster_size=cluster_size,
            assignment=assignment,
            total_packets=packets_for_fitness,
            seed=s,
        )
        scores.append(m["global_success"])
    return float(sum(scores) / len(scores))


def genetic_optimize_clusters(
    devices: List[Device],
    cluster_size: int,
    generations: int = 60,
    pop_size: int = 25,
    packets_for_fitness: int = 6000,
    traffic_seeds: Optional[List[int]] = None,
    per_gene_mutation_rate: Optional[float] = None,
    tournament_size: int = 3,
    elitism: int = 1,
    repair_balance: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    GA over per-device cluster assignments.

    Improvements:
    - Uniform crossover
    - Much smaller mutation rate by default for large N
    - Optional balance repair
    """
    if traffic_seeds is None:
        traffic_seeds = [101, 202]

    if NUM_CHANNELS % cluster_size != 0:
        raise ValueError("cluster_size must divide NUM_CHANNELS")

    N = len(devices)
    num_clusters = NUM_CHANNELS // cluster_size
    rng = random.Random(seed)

    if per_gene_mutation_rate is None:
        # good default for long chromosomes: ~2 mutations per individual on average
        per_gene_mutation_rate = min(0.02, 2.0 / max(N, 1))

    population = [[rng.randrange(num_clusters) for _ in range(N)] for _ in range(pop_size)]

    best_assignment = population[0][:]
    best_fit = -1e18
    fitness_history: List[float] = []

    for _gen in range(generations):
        fitnesses = [
            _evaluate_assignment(devices, indiv, cluster_size, packets_for_fitness, traffic_seeds)
            for indiv in population
        ]

        gen_best_i = int(np.argmax(fitnesses))
        if fitnesses[gen_best_i] > best_fit:
            best_fit = fitnesses[gen_best_i]
            best_assignment = population[gen_best_i][:]

        fitness_history.append(best_fit)

        new_pop: List[List[int]] = []
        for _ in range(elitism):
            new_pop.append(best_assignment[:])

        while len(new_pop) < pop_size:
            p1 = _tournament_selection(population, fitnesses, tournament_size, rng)
            p2 = _tournament_selection(population, fitnesses, tournament_size, rng)

            c1, c2 = _uniform_crossover(p1, p2, rng)
            _mutate(c1, num_clusters, per_gene_mutation_rate, rng)
            _mutate(c2, num_clusters, per_gene_mutation_rate, rng)

            if repair_balance:
                _repair_balance(c1, num_clusters, rng)
                _repair_balance(c2, num_clusters, rng)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    return {
        "best_assignment": best_assignment,
        "best_fitness": best_fit,
        "fitness_history": fitness_history,
        "cluster_size": cluster_size,
        "num_clusters": num_clusters,
        "N": len(devices),
    }


# -------------------------------
# Plotting
# -------------------------------
def plot_bars_success_and_collisions(results: Dict[str, Dict[str, Any]], out_prefix: str) -> None:
    order = list(results.keys())
    labels = [results[k]["label"] for k in order]
    succ = [results[k]["global_success"] for k in order]
    coll = [results[k]["collision_rate"] for k in order]

    x = range(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, succ)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Global success probability")
    plt.title("Success probability (fixed topology)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_success.png", dpi=200)
    print(f"Saved: {out_prefix}_success.png")

    plt.figure(figsize=(10, 5))
    plt.bar(x, coll)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Collision rate")
    plt.title("Collision rate (fixed topology)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_collisions.png", dpi=200)
    print(f"Saved: {out_prefix}_collisions.png")


def plot_success_cdf_vs_distance(results: Dict[str, Dict[str, Any]], out_path: str) -> None:
    """
    “CDF-like” curve: cumulative mean success within distance r.

    sort by distance:
      y(i) = mean success of devices with distance <= d(i)
    """
    plt.figure(figsize=(8, 6))
    for _key, m in results.items():
        d_m = m["per_device_distance_km"] * 1000.0
        p = m["per_device_success"]

        order = np.argsort(d_m)
        d_sorted = d_m[order]
        p_sorted = p[order]

        cum_mean = np.cumsum(p_sorted) / np.arange(1, len(p_sorted) + 1)
        plt.plot(d_sorted, cum_mean, label=m["label"])

    plt.xlabel("Distance to gateway (m)")
    plt.ylabel("Cumulative mean success within radius r")
    plt.title("CDF-style success vs distance (fixed topology)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


def plot_ga_fitness_histories(ga_runs: Dict[int, Dict[str, Any]], out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    for C, ga in ga_runs.items():
        hist = ga["fitness_history"]
        plt.plot(range(1, len(hist) + 1), hist, marker="o", label=f"GA C={C}")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (avg global success)")
    plt.title("GA fitness convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


# -------------------------------
# Experiments (fixed topology)
# -------------------------------
def run_all_algorithms_fixed_topology(
    N: int,
    topology_seed: int = 1,
    baseline_seed: int = 10,
    eval_seed: int = 999,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Runs baseline + heuristic + GA for C in {1,2,4} on THE SAME device placement.

    baseline_seed controls baseline traffic.
    eval_seed controls final evaluation traffic for clustered strategies.
    """
    devices = generate_devices(N, seed=topology_seed)

    results: Dict[str, Dict[str, Any]] = {}

    # Baseline
    m_base = run_simulation_with_devices(
        devices=devices,
        mode="baseline",
        cluster_size=1,
        assignment=None,
        total_packets=TOTAL_PACKETS,
        seed=baseline_seed,
    )
    m_base["label"] = "Baseline"
    results["baseline"] = m_base

    # Heuristic
    for C in (1, 2, 4):
        num_clusters = NUM_CHANNELS // C
        asg = assignment_distance_round_robin(devices, num_clusters)
        m = run_simulation_with_devices(
            devices=devices,
            mode="clustered",
            cluster_size=C,
            assignment=asg,
            total_packets=TOTAL_PACKETS,
            seed=eval_seed,
        )
        m["label"] = f"Heuristic C={C}"
        results[f"heur_C{C}"] = m

    # GA
    ga_runs: Dict[int, Dict[str, Any]] = {}
    for C in (1, 2, 4):
        ga = genetic_optimize_clusters(
            devices=devices,
            cluster_size=C,
            generations=60,
            pop_size=25,
            packets_for_fitness=6000,
            traffic_seeds=[101, 202],
            seed=123,
        )
        ga_runs[C] = ga

        # final evaluation with longer run
        m = run_simulation_with_devices(
            devices=devices,
            mode="clustered",
            cluster_size=C,
            assignment=ga["best_assignment"],
            total_packets=20_000,
            seed=eval_seed,
        )
        m["label"] = f"GA C={C}"
        m["fitness_history"] = ga["fitness_history"]
        results[f"ga_C{C}"] = m

    return results, ga_runs


def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    order = ["baseline", "heur_C1", "ga_C1", "heur_C2", "ga_C2", "heur_C4", "ga_C4"]
    for k in order:
        m = results[k]
        print(f"{m['label']:<14}  success={m['global_success']:.4f}  collisions={m['collision_rate']:.4f}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    N_FIXED = 10000

    t0 = time.time()
    results, ga_runs = run_all_algorithms_fixed_topology(
        N=N_FIXED,
        topology_seed=1,
        baseline_seed=10,
        eval_seed=999,
    )
    t1 = time.time()

    print_summary(results)
    print(f"\nTotal runtime: {t1 - t0:.2f} s")

    plot_bars_success_and_collisions(results, out_prefix=f"fixedN{N_FIXED}")
    plot_success_cdf_vs_distance(results, out_path=f"fixedN{N_FIXED}_cdf_success_vs_distance.png")
    plot_ga_fitness_histories(ga_runs, out_path=f"fixedN{N_FIXED}_ga_fitness.png")

    plt.show()
