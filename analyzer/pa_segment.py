"""Group pitch events into plate appearances by inter-pitch gap."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import PipelineConfig
from .pitch_detect import PitchEvent


@dataclass
class PlateAppearance:
    index: int  # 1-based ordinal within the inning
    pitches: list[PitchEvent] = field(default_factory=list)
    clip_start_s: float = 0.0
    clip_end_s: float = 0.0

    @property
    def first_pitch_s(self) -> float:
        return self.pitches[0].timestamp_s if self.pitches else self.clip_start_s

    @property
    def last_pitch_s(self) -> float:
        return self.pitches[-1].timestamp_s if self.pitches else self.clip_end_s

    @property
    def pitch_count(self) -> int:
        return len(self.pitches)


def group_into_pas(
    pitches: list[PitchEvent], config: PipelineConfig, video_duration_s: float
) -> list[PlateAppearance]:
    if not pitches:
        return []

    groups: list[list[PitchEvent]] = [[pitches[0]]]
    for p in pitches[1:]:
        gap = p.timestamp_s - groups[-1][-1].timestamp_s
        if gap > config.pa_max_gap_s:
            groups.append([p])
        else:
            groups[-1].append(p)

    pas: list[PlateAppearance] = []
    for i, grp in enumerate(groups):
        first = grp[0].timestamp_s
        last = grp[-1].timestamp_s
        start = max(0.0, first - config.pa_lead_s)
        end = min(video_duration_s, last + config.pa_trail_s)
        # Avoid clip overlap: don't extend past the next PA's start.
        if i + 1 < len(groups):
            next_first = groups[i + 1][0].timestamp_s
            end = min(end, max(last + 0.5, next_first - config.pa_lead_s))
        pas.append(
            PlateAppearance(
                index=i + 1, pitches=list(grp), clip_start_s=start, clip_end_s=end
            )
        )
    return pas
