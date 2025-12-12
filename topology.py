# topology.py
from dataclasses import dataclass
from typing import List, Optional
import math
import random

# --- Global constants from project spec ---

RADIUS_KM = 1.0                 # Cell radius (km)
TP_DBM = 5.0                    # Transmit power (dBm) for all devices
PACKET_DURATION = 0.05          # 50 ms
NUM_CHANNELS = 8
CAPTURE_THRESHOLD_DB = 3.0      # Capture rule threshold
MEAN_INTERARRIVAL_PER_DEVICE = 600.0  # seconds (per-device mean) :contentReference[oaicite:1]{index=1}
TOTAL_PACKETS = 10_000          # packets per simulation run :contentReference[oaicite:2]{index=2}


@dataclass
class Device:
    id: int
    x: float
    y: float
    distance: float      # distance to gateway in km
    rx_power: float      # received power in dBm
    cluster_id: int = -1


def generate_devices(N: int, rng: Optional[random.Random] = None) -> List[Device]:
    """
    Generate N devices uniformly over a disk of radius RADIUS_KM
    and compute their received power using the given path-loss model.
    """
    if rng is None:
        rng = random

    devices: List[Device] = []

    for i in range(N):
        # Uniform in area: r = R * sqrt(U), theta = 2πV
        u = rng.random()
        v = rng.random()
        r_km = RADIUS_KM * math.sqrt(u)
        theta = 2 * math.pi * v

        x = r_km * math.cos(theta)
        y = r_km * math.sin(theta)
        distance_km = r_km

        # Path-loss model uses distance in meters
        d_m = distance_km * 1000.0
        pl_db = 40.0 + 27.0 * math.log10(d_m)
        rx_dbm = TP_DBM - pl_db

        devices.append(
            Device(
                id=i,
                x=x,
                y=y,
                distance=distance_km,
                rx_power=rx_dbm
            )
        )

    return devices
