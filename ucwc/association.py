"""UE-to-BS association policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ucwc.channel_model import RadioLink
from ucwc.components import BaseStation, UE


@dataclass(frozen=True)
class ConnectionDecision:
    ue_id: str
    serving_bs_id: str | None
    is_connected: bool
    association_policy: str
    association_reason: str
    serving_sinr_db: float | None

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def associate_ues(
    base_stations: list[BaseStation],
    ues: list[UE],
    radio_links: list[RadioLink],
    policy: str = "best_sinr_with_capacity",
    allow_overload: bool = False,
) -> list[ConnectionDecision]:
    for bs in base_stations:
        bs.reset_connections()
    for ue in ues:
        ue.connected_bs_id = None

    bs_by_id = {bs.bs_id: bs for bs in base_stations}
    links_by_ue: dict[str, list[RadioLink]] = {}
    for link in radio_links:
        links_by_ue.setdefault(link.ue_id, []).append(link)

    decisions: list[ConnectionDecision] = []
    for ue in ues:
        candidates = sorted(
            links_by_ue.get(ue.ue_id, []),
            key=lambda link: link.sinr_db,
            reverse=True,
        )
        selected: RadioLink | None = None
        reason = "no_radio_link"
        for link in candidates:
            bs = bs_by_id[link.bs_id]
            if bs.can_attach():
                selected = link
                reason = "best_sinr_with_available_capacity"
                break
        if selected is None and candidates and allow_overload:
            selected = candidates[0]
            reason = "best_sinr_overloaded"

        if selected is None:
            decisions.append(
                ConnectionDecision(
                    ue_id=ue.ue_id,
                    serving_bs_id=None,
                    is_connected=False,
                    association_policy=policy,
                    association_reason=reason,
                    serving_sinr_db=None,
                )
            )
            continue

        bs = bs_by_id[selected.bs_id]
        attached = bs.attach(ue.ue_id)
        if not attached and allow_overload:
            bs.connected_ue_ids.append(ue.ue_id)
            attached = True
        ue.connected_bs_id = selected.bs_id if attached else None
        decisions.append(
            ConnectionDecision(
                ue_id=ue.ue_id,
                serving_bs_id=selected.bs_id if attached else None,
                is_connected=attached,
                association_policy=policy,
                association_reason=reason if attached else "capacity_blocked",
                serving_sinr_db=selected.sinr_db if attached else None,
            )
        )
    return decisions
