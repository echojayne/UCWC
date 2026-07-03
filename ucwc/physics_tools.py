"""Small physical proxy functions for the UCWC prototype."""

from __future__ import annotations

import math


def dbm_to_mw(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0)


def mw_to_dbm(mw: float) -> float:
    return 10.0 * math.log10(max(mw, 1e-15))


def thermal_noise_dbm(bandwidth_mhz: float, noise_figure_db: float = 7.0) -> float:
    bandwidth_hz = max(bandwidth_mhz, 1e-6) * 1_000_000.0
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def log_distance_pathloss_db(
    distance_m: float,
    carrier_frequency_ghz: float,
    pathloss_exponent: float,
    min_distance_m: float = 1.0,
) -> float:
    distance = max(distance_m, min_distance_m)
    fspl_1m_db = 32.4 + 20.0 * math.log10(max(carrier_frequency_ghz, 0.1))
    return fspl_1m_db + 10.0 * pathloss_exponent * math.log10(distance)


def sinr_db(
    serving_power_dbm: float,
    interference_powers_dbm: list[float],
    bandwidth_mhz: float,
    noise_figure_db: float,
) -> float:
    signal_mw = dbm_to_mw(serving_power_dbm)
    interference_mw = sum(dbm_to_mw(power) for power in interference_powers_dbm)
    noise_mw = dbm_to_mw(thermal_noise_dbm(bandwidth_mhz, noise_figure_db))
    return mw_to_dbm(signal_mw / max(interference_mw + noise_mw, 1e-15))


def sinr_to_cqi(value_db: float) -> int:
    thresholds = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7]
    cqi = 0
    for index, threshold in enumerate(thresholds, start=1):
        if value_db >= threshold:
            cqi = index
    return cqi


def spectral_efficiency_bps_hz(value_db: float) -> float:
    linear = 10.0 ** (value_db / 10.0)
    shannon = math.log2(1.0 + linear)
    return min(6.0, max(0.05, 0.55 * shannon))


def estimate_throughput_mbps(value_db: float, bandwidth_mhz: float) -> float:
    return round(max(0.0, spectral_efficiency_bps_hz(value_db) * bandwidth_mhz), 3)


def bandwidth_to_prb(bandwidth_mhz: float, cell_bandwidth_mhz: float, total_prb: int) -> int:
    if cell_bandwidth_mhz <= 0.0 or total_prb <= 0:
        return 0
    return max(1, int(round((bandwidth_mhz / cell_bandwidth_mhz) * total_prb)))


def required_bandwidth_mhz(min_dl_mbps: float, min_ul_mbps: float) -> float:
    return round(max(1.0, (min_dl_mbps + min_ul_mbps) / 4.0), 3)


def latency_proxy_ms(load_fraction: float, value_db: float, mobility_speed_kmh: float) -> float:
    radio_penalty = max(0.0, 8.0 - value_db) * 1.2
    load_penalty = 25.0 * max(0.0, load_fraction)
    mobility_penalty = min(18.0, mobility_speed_kmh / 12.0)
    return round(5.0 + radio_penalty + load_penalty + mobility_penalty, 3)


def packet_loss_proxy(value_db: float, load_fraction: float) -> float:
    radio_loss = 0.08 / (1.0 + math.exp((value_db - 1.0) / 2.0))
    load_loss = max(0.0, load_fraction - 0.75) * 0.08
    return round(min(0.25, radio_loss + load_loss), 5)


def reliability_proxy(packet_loss: float) -> float:
    return round(max(0.0, 1.0 - packet_loss), 5)


def jitter_proxy_ms(latency_ms: float, mobility_speed_kmh: float) -> float:
    return round(max(1.0, 0.12 * latency_ms + mobility_speed_kmh / 25.0), 3)
