"""Session-level UCWC configuration plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SessionConfigPlan:
    ue_id: str
    serving_bs_id: str
    bandwidth_quota_mhz: float
    qos_profile: str
    backup_bs_id: str | None = None
    handover_policy: str = "stable"
    security_profile: str = "standard"
    service_policy: str = "session_level_ucwc"
    source: str = "minimal_agent"

    def to_record(self) -> dict[str, object]:
        return asdict(self)
