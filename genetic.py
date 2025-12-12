# genetic.py
from typing import List, Tuple, Dict, Any, Optional
import random
import copy

import numpy as np

from topology import Device, generate_devices, NUM_CHANNELS
from simulation import run_simulation_with_devices


def evaluate_assignment(
    devices: List[Device],
    assignment: List[int],
    cluster_size: int,
    packets_for_fitness: int = 3000,
    fitness_seeds: Optional[List[int]] = None,
) -> float:
    """
    Apply a given cluster assignment to devices and run several shortened
    simulations (different seeds). Fitness = average global success probability.

    This reduces the randomness/noise compared to a single short run.
    """
    if fitness_seeds is None:
        fitness_seeds = [101]  # default: single seed

    num_clusters = NUM_CHANNELS // cluster_size
    if len(assignment) != len(devices):
        raise ValueError("assignment length != number of devices")
    for cid in assignment:
        if cid < 0 or cid >= num_clusters:
            raise ValueError("cluster id out of range in assignment")

    successes: List[float] = []

    for s in fitness_seeds:
        # Work on a deep copy so we don't permanently modify the original devices
        devs_copy: List[Device] = copy.deepcopy(devices)
        for dev, cid in zip(devs_copy, assignment):
            dev.cluster_id = cid

        metrics = run_simulation_with_devices(
            devices=devs_copy,
            mode="clustered",
            cluster_size=cluster_size,
            total_packets=packets_for_fitness,
            seed=s,
        )
        successes.append(metrics["global_success"])

    return float(sum(successes) / len(successes))


def tournament_selection(
    population: List[List[int]],
    fitnesses: List[float],
    tournament_size: int = 3,
) -> List[int]:
    """
    Tournament selection: pick 'tournament_size' random individuals and
    return the one with highest fitness.
    """
    best_idx = None
    best_fit = -1e9

    for _ in range(tournament_size):
        i = random.randrange(len(population))
        if best_idx is None or fitnesses[i] > best_fit:
            best_idx = i
            best_fit = fitnesses[i]

    return population[best_idx][:]  # return a copy


def one_point_crossover(
    parent1: List[int],
    parent2: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Simple one-point crossover.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have same length")
    n = len(parent1)
    if n <= 1:
        return parent1[:], parent2[:]

    point = random.randint(1, n - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(
    individual: List[int],
    num_clusters: int,
    mutation_rate: float = 0.01,
) -> None:
    """
    With probability = mutation_rate per gene, assign a new random cluster.
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randrange(num_clusters)


def genetic_optimize_clusters(
    N: int,
    cluster_size: int,
    generations: int = 150,
    pop_size: int = 20,
    packets_for_fitness: int = 3000,
    mutation_rate: float = 0.03,
    tournament_size: int = 3,
    seed: Optional[int] = None,
    fitness_seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run a GA to optimize cluster assignment for a given N and cluster_size.

    Heavier than before but higher quality:
    - more generations
    - larger population
    - fitness averaged over multiple seeds (if provided)

    Returns:
        {
          "best_assignment": [...],
          "best_fitness": float,
          "fitness_history": [...],
          "devices": devices   # the fixed topology
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if NUM_CHANNELS % cluster_size != 0:
        raise ValueError("cluster_size must divide NUM_CHANNELS")

    if fitness_seeds is None:
        # default: 2 different traffic seeds to reduce noise
        fitness_seeds = [101, 202]

    num_clusters = NUM_CHANNELS // cluster_size

    # Fixed topology for this GA run
    devices: List[Device] = generate_devices(N)

    # Initial population (random assignments)
    population: List[List[int]] = []
    for _ in range(pop_size):
        indiv = [random.randrange(num_clusters) for _ in range(N)]
        population.append(indiv)

    best_assignment: List[int] = population[0][:]
    best_fitness: float = -1e9
    fitness_history: List[float] = []

    for gen in range(generations):
        # Evaluate whole population
        fitnesses = [
            evaluate_assignment(
                devices,
                indiv,
                cluster_size,
                packets_for_fitness=packets_for_fitness,
                fitness_seeds=fitness_seeds,
            )
            for indiv in population
        ]

        # Track global best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_assignment = population[gen_best_idx][:]

        fitness_history.append(best_fitness)
        # IMPORTANT: no per-generation prints here to avoid log spam

        # New population via selection + crossover + mutation
        new_population: List[List[int]] = []

        # Elitism: keep the best individual
        new_population.append(best_assignment[:])

        # Fill the rest
        while len(new_population) < pop_size:
            # Select parents
            p1 = tournament_selection(population, fitnesses, tournament_size)
            p2 = tournament_selection(population, fitnesses, tournament_size)

            # Crossover
            child1, child2 = one_point_crossover(p1, p2)

            # Mutation
            mutate(child1, num_clusters, mutation_rate)
            mutate(child2, num_clusters, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    # Re-apply best assignment to devices (so caller can reuse them directly)
    for dev, cid in zip(devices, best_assignment):
        dev.cluster_id = cid

    result: Dict[str, Any] = {
        "best_assignment": best_assignment,
        "best_fitness": best_fitness,
        "fitness_history": fitness_history,
        "devices": devices,
        "cluster_size": cluster_size,
        "N": N,
    }
    return result
