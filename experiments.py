# experiments.py
from typing import Dict, Any

import matplotlib.pyplot as plt
import time

from simulation import run_simulation, run_simulation_with_devices
from clustering import assign_clusters_distance_based
from genetic import genetic_optimize_clusters


# Choose ONE N for all experiments
N_FIXED = 10_000  # you can change this (e.g. 20_000) if your machine can handle it


def run_all_algorithms_for_single_N(N: int) -> Dict[str, Dict[str, Any]]:
    """
    Run baseline + heuristic clustering (C=1,2,4) + GA (C=1,2,4) for a single N.
    Returns a dict:
        results["baseline"]
        results["heur_C1"], results["heur_C2"], results["heur_C4"]
        results["ga_C1"], results["ga_C2"], results["ga_C4"]
    """
    results: Dict[str, Dict[str, Any]] = {}

    print(f"\n==============================")
    print(f"=== MAIN EXPERIMENTS FOR N={N} ===")
    print(f"==============================\n")

    # 1) Baseline
    print("[Group 1: Baseline]")
    m_base = run_simulation(
        N=N,
        mode="baseline",
        cluster_size=1
    )
    m_base["label"] = "Baseline"
    results["baseline"] = m_base
    print(
        f"  [Baseline] success={m_base['global_success']:.4f}, "
        f"collision_rate={m_base['collision_rate']:.4f}"
    )

    # 2) Heuristic clustering for each C
    print("\n[Group 2: Heuristic Clustering (distance-based)]")
    for C in [1, 2, 4]:
        key = f"heur_C{C}"
        m_heur = run_simulation(
            N=N,
            mode="clustered",
            cluster_size=C,
            assign_clusters_fn=assign_clusters_distance_based,
        )
        m_heur["label"] = f"Heuristic C={C}"
        results[key] = m_heur
        print(
            f"  [Heuristic C={C}] success={m_heur['global_success']:.4f}, "
            f"collision_rate={m_heur['collision_rate']:.4f}"
        )

    # 3) GA clustering for each C (heavy, quality-focused)
    print("\n[Group 3: GA Clustering (optimized assignments)]")
    for C in [1, 2, 4]:
        print(f"  -- Running GA for cluster size C={C} (heavy, high-quality) --")
        ga_res = genetic_optimize_clusters(
            N=N,
            cluster_size=C,
            generations=150,          # strong GA, but not insane
            pop_size=20,
            packets_for_fitness=3000, # more packets -> less noise
            mutation_rate=0.03,
            seed=123,
            fitness_seeds=[101, 202], # average fitness over 2 traffic seeds
        )
        print(f"  [GA C={C}] best fitness (approx success, averaged) = {ga_res['best_fitness']:.4f}")

        # Final evaluation of GA best assignment with a long run
        devices = ga_res["devices"]
        best_assignment = ga_res["best_assignment"]
        for dev, cid in zip(devices, best_assignment):
            dev.cluster_id = cid

        m_ga = run_simulation_with_devices(
            devices=devices,
            mode="clustered",
            cluster_size=C,
            total_packets=20_000,   # long final run for smooth metrics
            seed=999,
        )
        m_ga["label"] = f"GA C={C}"
        # attach fitness history for GA convergence plots
        m_ga["fitness_history"] = ga_res["fitness_history"]
        results[f"ga_C{C}"] = m_ga

        print(
            f"  [GA C={C}] final success={m_ga['global_success']:.4f}, "
            f"collision_rate={m_ga['collision_rate']:.4f}"
        )

    return results


# ---------- Plot helpers ----------

def plot_bars_success_and_collisions(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Single bar chart summarising success & collisions for all algorithms.
    """
    labels = []
    success_vals = []
    coll_vals = []

    order = [
        "baseline",
        "heur_C1", "ga_C1",
        "heur_C2", "ga_C2",
        "heur_C4", "ga_C4",
    ]

    for key in order:
        m = results[key]
        labels.append(m["label"])
        success_vals.append(m["global_success"])
        coll_vals.append(m["collision_rate"])

    # Success
    plt.figure(figsize=(10, 5))
    x = range(len(labels))
    plt.bar(x, success_vals)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Global success probability")
    plt.title(f"Success probability for all algorithms (N={N_FIXED})")
    plt.tight_layout()
    plt.savefig("success_all_algorithms.png", dpi=200)
    print("Saved: success_all_algorithms.png")

    # Collisions
    plt.figure(figsize=(10, 5))
    plt.bar(x, coll_vals)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Collision rate")
    plt.title(f"Collision rate for all algorithms (N={N_FIXED})")
    plt.tight_layout()
    plt.savefig("collisions_all_algorithms.png", dpi=200)
    print("Saved: collisions_all_algorithms.png")


def plot_runtime_for_algorithms(N: int) -> None:
    """
    Measure and plot runtime vs 'algorithm variant' for a single N.
    You said you don't care about runtime; you can ignore this call.
    """
    runtimes: Dict[str, float] = {}

    print(f"\n==============================")
    print(f"=== RUNTIME MEASUREMENTS FOR N={N} ===")
    print(f"==============================\n")

    # Baseline
    print("[Runtime][Baseline]")
    t0 = time.time()
    run_simulation(N=N, mode="baseline", cluster_size=1)
    t1 = time.time()
    runtimes["Baseline"] = t1 - t0
    print(f"  Baseline runtime = {runtimes['Baseline']:.4f} s")

    # Heuristic C=1,2,4
    print("\n[Runtime][Heuristic Clustering]")
    for C in [1, 2, 4]:
        label = f"Heuristic C={C}"
        t0 = time.time()
        run_simulation(
            N=N,
            mode="clustered",
            cluster_size=C,
            assign_clusters_fn=assign_clusters_distance_based,
        )
        t1 = time.time()
        runtimes[label] = t1 - t0
        print(f"  {label} runtime = {runtimes[label]:.4f} s")

    # GA C=1,2,4 (runtime includes GA optimisation + final evaluation)
    print("\n[Runtime][GA Clustering]")
    for C in [1, 2, 4]:
        label = f"GA C={C}"
        print(f"  Measuring runtime for {label} ...")
        t0 = time.time()
        ga_res = genetic_optimize_clusters(
            N=N,
            cluster_size=C,
            generations=50,      # smaller here so runtime plot doesn't explode
            pop_size=10,
            packets_for_fitness=1500,
            mutation_rate=0.03,
            seed=42,
            fitness_seeds=[111],
        )
        devices = ga_res["devices"]
        best_assignment = ga_res["best_assignment"]
        for dev, cid in zip(devices, best_assignment):
            dev.cluster_id = cid
        run_simulation_with_devices(
            devices=devices,
            mode="clustered",
            cluster_size=C,
            total_packets=5_000,
            seed=99,
        )
        t1 = time.time()
        runtimes[label] = t1 - t0
        print(f"  {label} total runtime = {runtimes[label]:.4f} s")

    # Plot
    labels = list(runtimes.keys())
    vals = [runtimes[k] for k in labels]
    plt.figure(figsize=(10, 5))
    x = range(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Runtime (s)")
    plt.title(f"Simulation + GA runtime per algorithm (N={N})")
    plt.tight_layout()
    plt.savefig("runtime_all_algorithms.png", dpi=200)
    print("Saved: runtime_all_algorithms.png")


def plot_success_vs_distance(results: Dict[str, Dict[str, Any]], N: int) -> None:
    """
    Scatter plot success vs distance for baseline, heuristic (C=1,2,4), GA (C=1,2,4)
    all on one figure.
    """
    plt.figure(figsize=(8, 6))

    style_order = [
        "baseline",
        "heur_C1", "ga_C1",
        "heur_C2", "ga_C2",
        "heur_C4", "ga_C4",
    ]

    for key in style_order:
        m = results[key]
        d_m = m["per_device_distance"] * 1000.0
        p = m["per_device_success"]
        plt.scatter(d_m, p, alpha=0.3, label=m["label"])

    plt.xlabel("Distance to gateway (m)")
    plt.ylabel("Per-device success probability")
    plt.title(f"Success vs distance for all algorithms (N={N})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"success_vs_distance_all_N{N}.png", dpi=200)
    print(f"Saved: success_vs_distance_all_N{N}.png")


def plot_ga_fitness_histories(results: Dict[str, Dict[str, Any]], N: int) -> None:
    """
    Plot GA fitness history for each C on a single plot (so 'each cluster appears together').
    """
    plt.figure(figsize=(8, 6))

    for C in [1, 2, 4]:
        key = f"ga_C{C}"
        m = results[key]
        hist = m["fitness_history"]
        plt.plot(range(1, len(hist) + 1), hist, marker="o", label=f"GA C={C}")

    plt.xlabel("Generation")
    plt.ylabel("Best fitness (approx success)")
    plt.title(f"GA fitness vs generation for each clustering option (N={N})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ga_fitness_all_C_N{N}.png", dpi=200)
    print(f"Saved: ga_fitness_all_C_N{N}.png")


# ---------- Main ----------

if __name__ == "__main__":
    # 1) Run all algorithms (baseline + heuristic + GA for C=1,2,4)
    all_results = run_all_algorithms_for_single_N(N_FIXED)

    # 2) Plots for success/collisions vs algorithm
    plot_bars_success_and_collisions(all_results)

    # 3) Runtime vs algorithm (you can comment this out if you truly don't care)
    # plot_runtime_for_algorithms(N_FIXED)

    # 4) Success vs distance (all algorithms together)
    plot_success_vs_distance(all_results, N_FIXED)

    # 5) GA fitness vs generation for each cluster size, all on one figure
    plot_ga_fitness_histories(all_results, N_FIXED)

    # Show interactive windows if you want
    plt.show()
