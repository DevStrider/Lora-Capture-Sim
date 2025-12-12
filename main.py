import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

# Simulation parameters
RADIUS_KM = 1.0          # disk radius (km)
TP_DBM = 5.0             # transmit power (dBm)
NUM_CHANNELS = 8
PACKET_DURATION = 0.050  # seconds
TOTAL_PACKETS = 10_000
CAPTURE_THRESHOLD_DB = 3.0  # dB


# =========================
# Device & traffic generation
# =========================

def generate_devices(N: int, rng: np.random.Generator):
    """
    Generate N devices uniformly in area in a disk of radius 1 km,
    and compute their distance to gateway and RX power.

    Returns:
        distances_km: (N,) array of distances to gateway in km
        rx_dbm:       (N,) array of received powers in dBm
    """
    # Uniform in area: r = sqrt(U) * R
    U = rng.random(N)
    r_km = np.sqrt(U) * RADIUS_KM

    # theta uniform in [0, 2π) (we don't strictly need x,y here)
    theta = rng.random(N) * 2 * np.pi
    # x = r_km * np.cos(theta)
    # y = r_km * np.sin(theta)

    # Path loss and RX
    d_m = r_km * 1000.0
    pl_db = 40.0 + 27.0 * np.log10(d_m)
    rx_dbm = TP_DBM - pl_db

    return r_km, rx_dbm


def generate_events(N: int, total_packets: int, rng: np.random.Generator):
    """
    Generate (time, device_id) for each packet, using aggregate Poisson arrivals.
    Mean aggregate interarrival = 600 / N seconds.
    """
    delta_mean = 600.0 / N   # mean of exponential (seconds)

    # Exponential inter-arrivals with mean delta_mean
    interarrivals = rng.exponential(scale=delta_mean, size=total_packets)
    times = np.cumsum(interarrivals)

    # For each arrival, choose a random device (0..N-1)
    device_ids = rng.integers(low=0, high=N, size=total_packets)

    return times, device_ids


# =========================
# Core simulator
# =========================

def run_simulation(
    distances_km: np.ndarray,
    rx_dbm: np.ndarray,
    times: np.ndarray,
    device_ids: np.ndarray,
    rng: np.random.Generator,
    cluster_of_device: Optional[np.ndarray] = None,
    channels_per_cluster: Optional[int] = None,
):
    """
    General simulator with optional clustering.

    If cluster_of_device is None:
        - Each packet chooses a random channel in [0, NUM_CHANNELS-1].

    Else:
        - channels_per_cluster must be 1, 2, or 4.
        - NUM_CHANNELS must be divisible by channels_per_cluster.
        - Number of clusters K = NUM_CHANNELS // channels_per_cluster.
        - device i uses cluster c = cluster_of_device[i],
          then selects one of that cluster's channels uniformly.

    Returns:
        success_flags:    (P,) bool array indicating success of each packet
        packet_distances: (P,) array with distance of the transmitting device
    """
    P = len(times)
    assert P == len(device_ids)

    # Active packets per channel: list of (end_time, rx_dbm)
    active = [[] for _ in range(NUM_CHANNELS)]

    success_flags = np.zeros(P, dtype=bool)
    packet_distances = np.zeros(P, dtype=float)

    # Sanity checks for clustering mode
    if cluster_of_device is not None:
        assert channels_per_cluster in (1, 2, 4)
        assert NUM_CHANNELS % channels_per_cluster == 0
        num_clusters = NUM_CHANNELS // channels_per_cluster
        assert cluster_of_device.shape[0] == rx_dbm.shape[0]

    for k in range(P):
        t = times[k]
        dev = int(device_ids[k])
        rx = float(rx_dbm[dev])

        # Record distance of this packet's device (for analysis/plots)
        packet_distances[k] = float(distances_km[dev])

        # === Choose channel ===
        if cluster_of_device is None:
            # Baseline: uniform random over all channels
            ch = int(rng.integers(low=0, high=NUM_CHANNELS))
        else:
            # Clustered: device -> cluster -> subset of channels
            c = int(cluster_of_device[dev])  # cluster index
            start_ch = c * channels_per_cluster
            end_ch = start_ch + channels_per_cluster  # exclusive
            ch = int(rng.integers(low=start_ch, high=end_ch))

        # === Remove expired packets from this channel ===
        channel_list = active[ch]
        channel_list = [pkt for pkt in channel_list if pkt[0] > t]
        active[ch] = channel_list

        # === Collision / capture ===
        overlapping_rx = [rx] + [pkt[1] for pkt in channel_list]

        if len(overlapping_rx) == 1:
            # No collision
            success_flags[k] = True
        else:
            max_rx = max(overlapping_rx)
            count_max = sum(1 for v in overlapping_rx if v == max_rx)

            if count_max == 1:
                sorted_rx = sorted(overlapping_rx, reverse=True)
                second_max = sorted_rx[1]
                if max_rx - second_max >= CAPTURE_THRESHOLD_DB:
                    # Strongest wins, check if new packet is strongest
                    if rx == max_rx:
                        success_flags[k] = True
                    # else: some other overlapping packet wins, this one fails
                else:
                    # capture not satisfied -> all fail
                    success_flags[k] = False
            else:
                # Tie for maximum -> all fail
                success_flags[k] = False

        # Add new packet to active list
        end_time = t + PACKET_DURATION
        active[ch].append((end_time, rx))

    return success_flags, packet_distances


# =========================
# Analysis: success vs distance
# =========================

def compute_success_vs_distance(
    packet_distances: np.ndarray,
    success_flags: np.ndarray,
    num_bins: int = 10,
):
    """
    Compute success probability as a function of distance by binning.

    Returns:
        bin_centers:  (num_bins,) array of bin centers (km)
        success_prob: (num_bins,) array, NaN for empty bins
    """
    assert packet_distances.shape == success_flags.shape

    bins = np.linspace(0.0, RADIUS_KM, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    success_prob = np.full(num_bins, np.nan, dtype=float)

    for i in range(num_bins):
        mask = (packet_distances >= bins[i]) & (packet_distances < bins[i + 1])
        if np.any(mask):
            success_prob[i] = success_flags[mask].mean()

    return bin_centers, success_prob


# =========================
# Clustering heuristics
# =========================

def distance_based_clustering(distances_km: np.ndarray, channels_per_cluster: int) -> np.ndarray:
    """
    Simple distance-based clustering:
    - Sort devices by distance (ascending).
    - Split them into K = NUM_CHANNELS // channels_per_cluster groups.
    - Assign each group to one cluster index.

    Returns:
        cluster_of_device: (N,) int array with values in [0, K-1]
    """
    assert channels_per_cluster in (1, 2, 4)
    assert NUM_CHANNELS % channels_per_cluster == 0

    N = distances_km.shape[0]
    num_clusters = NUM_CHANNELS // channels_per_cluster

    # Order devices from closest to farthest
    order = np.argsort(distances_km)
    cluster_of_device = np.empty(N, dtype=int)

    # Split into num_clusters contiguous segments in this order
    segment_size = N // num_clusters
    remainder = N % num_clusters  # distribute extra devices

    start = 0
    for c in range(num_clusters):
        extra = 1 if c < remainder else 0
        end = start + segment_size + extra
        idx_segment = order[start:end]
        cluster_of_device[idx_segment] = c
        start = end

    return cluster_of_device


def greedy_clustering(rx_dbm: np.ndarray, channels_per_cluster: int) -> np.ndarray:
    """
    Greedy clustering based on received power:
    - Convert RX from dBm to linear power.
    - Sort devices by descending RX (strongest first).
    - Maintain a 'load' per cluster = sum of linear power of devices in that cluster.
    - Assign each device to the cluster with the smallest current load.

    Returns:
        cluster_of_device: (N,) int array with values in [0, K-1]
    """
    assert channels_per_cluster in (1, 2, 4)
    assert NUM_CHANNELS % channels_per_cluster == 0

    N = rx_dbm.shape[0]
    num_clusters = NUM_CHANNELS // channels_per_cluster

    # Convert dBm to linear power (relative units)
    power_lin = 10.0 ** (rx_dbm / 10.0)

    # Order devices from strongest to weakest
    order = np.argsort(-rx_dbm)  # descending

    cluster_of_device = np.empty(N, dtype=int)
    cluster_loads = np.zeros(num_clusters, dtype=float)

    for dev in order:
        c = int(np.argmin(cluster_loads))  # cluster with smallest current load
        cluster_of_device[dev] = c
        cluster_loads[c] += power_lin[dev]

    return cluster_of_device


# =========================
# Genetic Algorithm for clustering
# =========================

def evaluate_clustering_fitness(
    cluster_of_device: np.ndarray,
    channels_per_cluster: int,
    distances_km: np.ndarray,
    rx_dbm: np.ndarray,
    times: np.ndarray,
    device_ids: np.ndarray,
) -> float:
    """
    Fitness = global success probability for a given clustering.
    Uses a fresh RNG so each evaluation has its own random channel choices.
    """
    rng_sim = np.random.default_rng()
    success_flags, _ = run_simulation(
        distances_km=distances_km,
        rx_dbm=rx_dbm,
        times=times,
        device_ids=device_ids,
        rng=rng_sim,
        cluster_of_device=cluster_of_device,
        channels_per_cluster=channels_per_cluster,
    )
    return float(success_flags.mean())


def tournament_select(fitnesses: np.ndarray, rng: np.random.Generator, k: int = 3) -> int:
    """
    Tournament selection: pick k random individuals, return index of the best one.
    """
    pop_size = len(fitnesses)
    idxs = rng.integers(low=0, high=pop_size, size=k)
    best_idx = idxs[0]
    best_fit = fitnesses[best_idx]
    for i in idxs[1:]:
        if fitnesses[i] > best_fit:
            best_idx = i
            best_fit = fitnesses[i]
    return int(best_idx)


def crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator):
    """
    Single-point crossover.
    """
    N = parent1.shape[0]
    point = int(rng.integers(low=1, high=N))  # cut between 1 and N-1
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    child1[:point] = parent1[:point]
    child1[point:] = parent2[point:]
    child2[:point] = parent2[:point]
    child2[point:] = parent1[point:]
    return child1, child2


def mutate(chrom: np.ndarray, mutation_rate: float, num_clusters: int, rng: np.random.Generator):
    """
    Mutate chromosome in-place: with probability mutation_rate,
    change the cluster of each device to a different random cluster.
    """
    N = chrom.shape[0]
    for i in range(N):
        if rng.random() < mutation_rate:
            old = chrom[i]
            new = int(rng.integers(low=0, high=num_clusters))
            # ensure a change
            while new == old:
                new = int(rng.integers(low=0, high=num_clusters))
            chrom[i] = new


def ga_optimize_clustering(
    distances_km: np.ndarray,
    rx_dbm: np.ndarray,
    times: np.ndarray,
    device_ids: np.ndarray,
    channels_per_cluster: int,
    rng_ga: np.random.Generator,
    pop_size: int = 12,
    num_generations: int = 30,
    mutation_rate: float = 0.01,
):
    """
    Genetic Algorithm to optimize cluster_of_device for a fixed channels_per_cluster.

    Returns:
        best_chrom:  best clustering found (N,)
        best_fit:    its fitness (global success probability)
    """
    assert channels_per_cluster in (1, 2, 4)
    num_clusters = NUM_CHANNELS // channels_per_cluster
    N = rx_dbm.shape[0]

    # --- Initial population: distance-based, greedy, plus random ---
    population = []

    # Seed 1: distance-based
    chrom_dist = distance_based_clustering(distances_km, channels_per_cluster)
    population.append(chrom_dist)

    # Seed 2: greedy
    chrom_greedy = greedy_clustering(rx_dbm, channels_per_cluster)
    population.append(chrom_greedy)

    # Fill the rest with random clusterings
    while len(population) < pop_size:
        chrom_rand = rng_ga.integers(low=0, high=num_clusters, size=N, dtype=int)
        population.append(chrom_rand)

    population = np.array(population, dtype=int)  # shape (pop_size, N)

    # --- Evaluate initial fitness ---
    fitnesses = np.zeros(pop_size, dtype=float)
    for i in range(pop_size):
        fitnesses[i] = evaluate_clustering_fitness(
            population[i],
            channels_per_cluster,
            distances_km,
            rx_dbm,
            times,
            device_ids,
        )

    # --- GA main loop ---
    for gen in range(num_generations):
        new_population = []

        # Elitism: keep the best individual
        best_idx = int(np.argmax(fitnesses))
        best_chrom = population[best_idx].copy()
        best_fit = fitnesses[best_idx]
        new_population.append(best_chrom)

        # Fill the rest of the population
        while len(new_population) < pop_size:
            # Select parents
            p1_idx = tournament_select(fitnesses, rng_ga, k=3)
            p2_idx = tournament_select(fitnesses, rng_ga, k=3)
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # Crossover (with high probability)
            if rng_ga.random() < 0.8:
                child1, child2 = crossover(parent1, parent2, rng_ga)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # Mutation
            mutate(child1, mutation_rate, num_clusters, rng_ga)
            mutate(child2, mutation_rate, num_clusters, rng_ga)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = np.array(new_population, dtype=int)

        # Recompute fitnesses
        for i in range(pop_size):
            fitnesses[i] = evaluate_clustering_fitness(
                population[i],
                channels_per_cluster,
                distances_km,
                rx_dbm,
                times,
                device_ids,
            )

    # Final best
    best_idx = int(np.argmax(fitnesses))
    best_chrom = population[best_idx].copy()
    best_fit = fitnesses[best_idx]
    return best_chrom, best_fit


# =========================
# Experiment harness
# =========================

def run_single_scenario_debug():
    """
    Original single-scenario run for N=200, useful for debugging and understanding.
    """
    N = 200
    rng = np.random.default_rng()

    distances_km, rx_dbm = generate_devices(N, rng)
    times, device_ids = generate_events(N, TOTAL_PACKETS, rng)

    # Baseline
    rng_sim = np.random.default_rng()
    success_flags_base, packet_distances_base = run_simulation(
        distances_km=distances_km,
        rx_dbm=rx_dbm,
        times=times,
        device_ids=device_ids,
        rng=rng_sim,
        cluster_of_device=None,
        channels_per_cluster=None,
    )
    global_success_base = success_flags_base.mean()
    print(f"Baseline (random channels) global success probability: {global_success_base:.4f}")

    bin_centers, success_prob = compute_success_vs_distance(packet_distances_base, success_flags_base)
    print("Baseline distance bin centers (km):", bin_centers)
    print("Baseline success probability per bin:", success_prob)

    # Distance-based
    for C in (1, 2, 4):
        cluster_dist = distance_based_clustering(distances_km, channels_per_cluster=C)
        rng_sim_dist = np.random.default_rng()
        success_flags_dist, _ = run_simulation(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            rng=rng_sim_dist,
            cluster_of_device=cluster_dist,
            channels_per_cluster=C,
        )
        print(f"Distance-based clustering (C={C}) global success: {success_flags_dist.mean():.4f}")

    # Greedy
    for C in (1, 2, 4):
        cluster_greedy = greedy_clustering(rx_dbm, channels_per_cluster=C)
        rng_sim_greedy = np.random.default_rng()
        success_flags_greedy, _ = run_simulation(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            rng=rng_sim_greedy,
            cluster_of_device=cluster_greedy,
            channels_per_cluster=C,
        )
        print(f"Greedy clustering (C={C}) global success: {success_flags_greedy.mean():.4f}")

    # GA
    for C in (1, 2, 4):
        rng_ga = np.random.default_rng()
        best_chrom, best_fit = ga_optimize_clustering(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            channels_per_cluster=C,
            rng_ga=rng_ga,
            pop_size=12,
            num_generations=30,
            mutation_rate=0.01,
        )
        print(f"GA-optimized clustering (C={C}) best global success: {best_fit:.4f}")


def run_all_algorithms_for_N(
    N: int,
    channels_per_cluster_values=(1, 2, 4),
) -> List[Dict]:
    """
    Run baseline, distance-based, greedy, and GA clustering for a given N.
    Returns a list of result dicts with:
        N, algo, C, success_prob, collisions, runtime
    """
    results = []

    # RNG for devices & events for this N
    rng = np.random.default_rng()
    distances_km, rx_dbm = generate_devices(N, rng)
    times, device_ids = generate_events(N, TOTAL_PACKETS, rng)

    # --- Baseline ---
    rng_sim = np.random.default_rng()
    start = time.perf_counter()
    success_flags, _ = run_simulation(
        distances_km=distances_km,
        rx_dbm=rx_dbm,
        times=times,
        device_ids=device_ids,
        rng=rng_sim,
        cluster_of_device=None,
        channels_per_cluster=None,
    )
    end = time.perf_counter()
    success_prob = success_flags.mean()
    collisions = TOTAL_PACKETS - success_flags.sum()
    results.append({
        "N": N,
        "algo": "baseline",
        "C": None,
        "success_prob": success_prob,
        "collisions": int(collisions),
        "runtime": end - start,
    })

    # --- Distance-based & Greedy & GA for each C ---
    for C in channels_per_cluster_values:
        # Distance-based
        cluster_dist = distance_based_clustering(distances_km, channels_per_cluster=C)
        rng_sim_dist = np.random.default_rng()
        start = time.perf_counter()
        success_flags_dist, _ = run_simulation(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            rng=rng_sim_dist,
            cluster_of_device=cluster_dist,
            channels_per_cluster=C,
        )
        end = time.perf_counter()
        results.append({
            "N": N,
            "algo": "distance",
            "C": C,
            "success_prob": success_flags_dist.mean(),
            "collisions": int(TOTAL_PACKETS - success_flags_dist.sum()),
            "runtime": end - start,
        })

        # Greedy
        cluster_greedy = greedy_clustering(rx_dbm, channels_per_cluster=C)
        rng_sim_greedy = np.random.default_rng()
        start = time.perf_counter()
        success_flags_greedy, _ = run_simulation(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            rng=rng_sim_greedy,
            cluster_of_device=cluster_greedy,
            channels_per_cluster=C,
        )
        end = time.perf_counter()
        results.append({
            "N": N,
            "algo": "greedy",
            "C": C,
            "success_prob": success_flags_greedy.mean(),
            "collisions": int(TOTAL_PACKETS - success_flags_greedy.sum()),
            "runtime": end - start,
        })

        # GA (includes GA optimization time + one final simulation)
        rng_ga = np.random.default_rng()
        start = time.perf_counter()
        best_chrom, best_fit = ga_optimize_clustering(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            channels_per_cluster=C,
            rng_ga=rng_ga,
            pop_size=12,
            num_generations=30,
            mutation_rate=0.01,
        )
        rng_sim_ga = np.random.default_rng()
        success_flags_ga, _ = run_simulation(
            distances_km=distances_km,
            rx_dbm=rx_dbm,
            times=times,
            device_ids=device_ids,
            rng=rng_sim_ga,
            cluster_of_device=best_chrom,
            channels_per_cluster=C,
        )
        end = time.perf_counter()

        results.append({
            "N": N,
            "algo": "ga",
            "C": C,
            "success_prob": success_flags_ga.mean(),
            "collisions": int(TOTAL_PACKETS - success_flags_ga.sum()),
            "runtime": end - start,
        })

    return results


def run_experiments():
    """
    Sweep over N values and collect success, collisions, and runtime
    for baseline, distance-based, greedy, and GA algorithms.
    """
    N_VALUES = [50, 100, 150, 200, 300]  # adjust as needed
    all_results: List[Dict] = []

    for N in N_VALUES:
        print(f"\n=== Running experiments for N={N} devices ===")
        res_N = run_all_algorithms_for_N(N)
        all_results.extend(res_N)
        for r in res_N:
            print(f"  {r['algo']:8s} C={str(r['C']):>4s}  "
                  f"success={r['success_prob']:.4f}  "
                  f"collisions={r['collisions']:4d}  "
                  f"runtime={r['runtime']:.4f} s")

    # Helper to filter results
    def filter_results(algo: str, C: Optional[int]):
        xs, succ, col, rt = [], [], [], []
        for r in all_results:
            if r["algo"] == algo and r["C"] == C:
                xs.append(r["N"])
                succ.append(r["success_prob"])
                col.append(r["collisions"])
                rt.append(r["runtime"])
        xs = np.array(xs)
        succ = np.array(succ)
        col = np.array(col)
        rt = np.array(rt)
        # sort by N
        order = np.argsort(xs)
        return xs[order], succ[order], col[order], rt[order]

    # Example plots: success vs N for baseline, greedy C=1, GA C=1
    Ns_base, succ_base, col_base, rt_base = filter_results("baseline", None)
    Ns_greedy1, succ_greedy1, col_greedy1, rt_greedy1 = filter_results("greedy", 1)
    Ns_ga1, succ_ga1, col_ga1, rt_ga1 = filter_results("ga", 1)

    plt.figure()
    plt.plot(Ns_base, succ_base, marker='o', label='Baseline')
    plt.plot(Ns_greedy1, succ_greedy1, marker='s', label='Greedy C=1')
    plt.plot(Ns_ga1, succ_ga1, marker='^', label='GA C=1')
    plt.xlabel('Number of devices N')
    plt.ylabel('Success probability')
    plt.ylim(0.95, 1.01)
    plt.grid(True)
    plt.legend()
    plt.title('Success probability vs N')
    plt.tight_layout()

    # Collisions vs N
    plt.figure()
    plt.plot(Ns_base, col_base, marker='o', label='Baseline')
    plt.plot(Ns_greedy1, col_greedy1, marker='s', label='Greedy C=1')
    plt.plot(Ns_ga1, col_ga1, marker='^', label='GA C=1')
    plt.xlabel('Number of devices N')
    plt.ylabel('Number of collisions (out of 10,000 packets)')
    plt.grid(True)
    plt.legend()
    plt.title('Collisions vs N')
    plt.tight_layout()

    # Runtime vs N
    plt.figure()
    plt.plot(Ns_base, rt_base, marker='o', label='Baseline')
    plt.plot(Ns_greedy1, rt_greedy1, marker='s', label='Greedy C=1')
    plt.plot(Ns_ga1, rt_ga1, marker='^', label='GA C=1')
    plt.xlabel('Number of devices N')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.legend()
    plt.title('Runtime vs N')
    plt.tight_layout()

    plt.show()


# =========================
# Main
# =========================

def main():
    # First, run the single-scenario debug (N=200) to see detailed behavior
    run_single_scenario_debug()

    # Then, run experiments vs N for plots
    run_experiments()


if __name__ == "__main__":
    main()
