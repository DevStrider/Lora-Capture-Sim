# simulation.py
from typing import List, Dict, Any, Optional, Callable
import random

import numpy as np

from topology import (
    Device,
    generate_devices,
    MEAN_INTERARRIVAL_PER_DEVICE,
    PACKET_DURATION,
    NUM_CHANNELS,
    CAPTURE_THRESHOLD_DB,
    TOTAL_PACKETS,
)
from clustering import build_cluster_channel_map, assign_clusters_random


def evaluate_collisions_and_success(packets: List[Dict[str, Any]]) -> None:
    """
    Given a list of packets with:
        - channel
        - start_time
        - end_time
        - rx  (received power in dBm)
    fill in:
        - success: bool
        - collided: bool

    Logic (per channel):
    - Build groups of packets whose time intervals overlap (transitively).
    - For each group:
        * if size == 1: that packet succeeds.
        * else:
            - Find strongest rx.
            - If a unique strongest, and it's >= 3 dB stronger than all others:
                that one succeeds, the rest collide.
            - Otherwise: all collide.
    """
    # Initialize flags
    for p in packets:
        p["success"] = False
        p["collided"] = False

    # Group packet indices by channel
    by_channel: Dict[int, List[int]] = {}
    for idx, p in enumerate(packets):
        ch = p["channel"]
        by_channel.setdefault(ch, []).append(idx)

    # Process each channel separately
    for ch, idx_list in by_channel.items():
        # Sort by start time
        idx_list_sorted = sorted(idx_list, key=lambda i: packets[i]["start_time"])
        if not idx_list_sorted:
            continue

        current_group = [idx_list_sorted[0]]
        group_end = packets[idx_list_sorted[0]]["end_time"]

        def process_group(group_indices: List[int]) -> None:
            if not group_indices:
                return

            # Single packet -> always success
            if len(group_indices) == 1:
                i = group_indices[0]
                packets[i]["success"] = True
                packets[i]["collided"] = False
                return

            # Multiple overlapping packets: apply capture rule
            rx_vals = np.array([packets[i]["rx"] for i in group_indices])
            max_rx = float(rx_vals.max())
            tol = 1e-9

            # Find all packets with max_rx (allowing small numeric tolerance)
            max_positions = [
                pos for pos, i in enumerate(group_indices)
                if abs(packets[i]["rx"] - max_rx) <= tol
            ]

            # Tie for strongest -> nobody captured, all collide
            if len(max_positions) != 1:
                for i in group_indices:
                    packets[i]["success"] = False
                    packets[i]["collided"] = True
                return

            winner_pos = max_positions[0]
            winner_idx = group_indices[winner_pos]

            # Check 3 dB condition
            capture_ok = True
            for pos, i in enumerate(group_indices):
                if pos == winner_pos:
                    continue
                if max_rx - packets[i]["rx"] < CAPTURE_THRESHOLD_DB:
                    capture_ok = False
                    break

            if capture_ok:
                # Winner succeeds, others collide
                for pos, i in enumerate(group_indices):
                    if pos == winner_pos:
                        packets[i]["success"] = True
                        packets[i]["collided"] = False
                    else:
                        packets[i]["success"] = False
                        packets[i]["collided"] = True
            else:
                # Everyone loses
                for i in group_indices:
                    packets[i]["success"] = False
                    packets[i]["collided"] = True

        # Sweep to form overlap groups (transitive closure)
        for idx in idx_list_sorted[1:]:
            p = packets[idx]
            if p["start_time"] < group_end:
                # Overlaps with current group
                current_group.append(idx)
                if p["end_time"] > group_end:
                    group_end = p["end_time"]
            else:
                # No overlap: finish previous group, start new one
                process_group(current_group)
                current_group = [idx]
                group_end = p["end_time"]

        # Process last group
        process_group(current_group)


def _simulate_core(
    devices: List[Device],
    mode: str,
    cluster_size: int,
    total_packets: int,
) -> Dict[str, Any]:
    """
    Core simulation given a fixed list of devices (with cluster_id already set
    when mode='clustered').
    """
    if mode not in ("baseline", "clustered"):
        raise ValueError("mode must be 'baseline' or 'clustered'")

    cluster_channels: Optional[Dict[int, List[int]]] = None
    num_clusters: Optional[int] = None

    if mode == "clustered":
        if NUM_CHANNELS % cluster_size != 0:
            raise ValueError("cluster_size must divide NUM_CHANNELS")
        num_clusters = NUM_CHANNELS // cluster_size

        # Sanity check cluster ids
        for d in devices:
            if d.cluster_id < 0 or d.cluster_id >= num_clusters:
                raise ValueError("Device has invalid cluster_id. Did you assign clusters?")

        cluster_channels = build_cluster_channel_map(cluster_size)
    else:
        # baseline: cluster_id not used
        for d in devices:
            d.cluster_id = 0

    # 1) Generate packet transmissions
    packets: List[Dict[str, Any]] = []

    N = len(devices)
    time_now = 0.0
    delta_mean = MEAN_INTERARRIVAL_PER_DEVICE / N

    for _ in range(total_packets):
        inter_arrival = np.random.exponential(delta_mean)
        time_now += inter_arrival

        dev = random.choice(devices)
        dev_id = dev.id

        if mode == "baseline":
            channel = random.randint(0, NUM_CHANNELS - 1)
        else:
            assert cluster_channels is not None
            ch_list = cluster_channels[dev.cluster_id]
            channel = random.choice(ch_list)

        packets.append(
            {
                "device_id": dev_id,
                "channel": channel,
                "start_time": time_now,
                "end_time": time_now + PACKET_DURATION,
                "rx": dev.rx_power,
                # success/collided filled later
            }
        )

    # 2) Apply capture rule & collisions
    evaluate_collisions_and_success(packets)

    # 3) Aggregate metrics
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

    distances = np.array([d.distance for d in devices])
    per_device_success = np.zeros(N)
    for i in range(N):
        if sent[i] > 0:
            per_device_success[i] = succ[i] / sent[i]
        else:
            per_device_success[i] = 0.0

    global_success = total_success / total_packets
    collision_rate = total_collided / total_packets

    metrics: Dict[str, Any] = {
        "mode": mode,
        "N": N,
        "cluster_size": cluster_size,
        "global_success": global_success,
        "collision_rate": collision_rate,
        "per_device_distance": distances,
        "per_device_success": per_device_success,
        "packets": packets,
    }
    if num_clusters is not None:
        metrics["num_clusters"] = num_clusters

    return metrics


def run_simulation(
    N: int,
    mode: str = "baseline",   # "baseline" or "clustered"
    cluster_size: int = 1,
    assign_clusters_fn: Optional[Callable[[List[Device], int], None]] = None,
    custom_assignment: Optional[List[int]] = None,
    total_packets: int = TOTAL_PACKETS,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-level convenience wrapper that:
        - generates devices
        - assigns clusters (if clustered)
        - calls the core simulator

    For GA with fixed topology, use run_simulation_with_devices().
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    devices = generate_devices(N)

    if mode == "clustered":
        if NUM_CHANNELS % cluster_size != 0:
            raise ValueError("cluster_size must divide NUM_CHANNELS")
        num_clusters = NUM_CHANNELS // cluster_size

        if custom_assignment is not None:
            if len(custom_assignment) != N:
                raise ValueError("custom_assignment length must equal N")
            for dev, cid in zip(devices, custom_assignment):
                if cid < 0 or cid >= num_clusters:
                    raise ValueError("cluster id out of range in custom_assignment")
                dev.cluster_id = cid
        elif assign_clusters_fn is not None:
            assign_clusters_fn(devices, num_clusters)
        else:
            assign_clusters_random(devices, num_clusters)
    elif mode == "baseline":
        for dev in devices:
            dev.cluster_id = 0
    else:
        raise ValueError("Unknown mode: must be 'baseline' or 'clustered'")

    metrics = _simulate_core(
        devices=devices,
        mode=mode,
        cluster_size=cluster_size,
        total_packets=total_packets,
    )
    return metrics


def run_simulation_with_devices(
    devices: List[Device],
    mode: str,
    cluster_size: int,
    total_packets: int = TOTAL_PACKETS,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lower-level function for GA:

    - devices are already generated (fixed topology)
    - cluster_id for each device is already set (for clustered mode)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    return _simulate_core(
        devices=devices,
        mode=mode,
        cluster_size=cluster_size,
        total_packets=total_packets,
    )
