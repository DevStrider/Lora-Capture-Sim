# clustering.py
from typing import List, Dict, Optional
import random

from topology import Device, NUM_CHANNELS


def build_cluster_channel_map(cluster_size: int) -> Dict[int, List[int]]:
    """
    Split the 8 channels into disjoint groups (clusters) of size cluster_size.

    Example:
        cluster_size = 2  -> 4 clusters, each with 2 channels.
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


def assign_clusters_distance_based(devices: List[Device], num_clusters: int) -> None:
    """
    Simple heuristic:
    - Sort devices by distance (ascending)
    - Assign them round-robin to clusters 0..num_clusters-1
    """
    devices_sorted = sorted(devices, key=lambda d: d.distance)

    for idx, dev in enumerate(devices_sorted):
        dev.cluster_id = idx % num_clusters


def assign_clusters_random(
    devices: List[Device],
    num_clusters: int,
    rng: Optional[random.Random] = None
) -> None:
    """
    Assign each device randomly to one of the num_clusters clusters.
    """
    if rng is None:
        rng = random

    for dev in devices:
        dev.cluster_id = rng.randint(0, num_clusters - 1)
