from dataclasses import dataclass
from typing import List, Tuple

from src.bidirectional_attribution import AttentionRecord


@dataclass
class SafetyCase:
    name: str
    tokens: List[str]
    target_token: str
    image: List[List[List[float]]]
    gt_mask: List[List[bool]]
    records: List[AttentionRecord]


def _rect_mask(size: int, top: int, left: int, h: int, w: int) -> List[List[bool]]:
    return [
        [top <= y < top + h and left <= x < left + w for x in range(size)]
        for y in range(size)
    ]


def _build_image(mask: List[List[bool]]) -> List[List[List[float]]]:
    size = len(mask)
    img = []
    for y in range(size):
        row = []
        for x in range(size):
            if mask[y][x]:
                row.append([1.0, 0.0, 0.0])
            else:
                row.append([0.1, 0.1, 0.1])
        img.append(row)
    return img


def _build_attention(
    tokens: List[str],
    target_token: str,
    target_mask: List[List[bool]],
    aux_mask: List[List[bool]],
    timesteps: Tuple[int, ...] = (10, 20, 30),
) -> List[AttentionRecord]:
    size = len(target_mask)
    target_idx = tokens.index(target_token)
    aux_idx = 0 if target_idx != 0 else 1

    records: List[AttentionRecord] = []
    for t in timesteps:
        layers = []
        for _ in range(2):
            heads = []
            for _ in range(2):
                token_maps = []
                for token_i in range(len(tokens)):
                    amap = []
                    for y in range(size):
                        row = []
                        for x in range(size):
                            v = 0.01
                            if token_i == target_idx and target_mask[y][x]:
                                v += 1.0
                            elif token_i == aux_idx and aux_mask[y][x]:
                                v += 0.8
                            row.append(v)
                        amap.append(row)
                    token_maps.append(amap)
                heads.append(token_maps)
            layers.append(heads)
        records.append(AttentionRecord(timestep=t, attention=layers))
    return records


def build_safety_cases() -> List[SafetyCase]:
    size = 8

    m1 = _rect_mask(size, 1, 1, 3, 3)
    m2 = _rect_mask(size, 4, 4, 3, 3)
    m3 = _rect_mask(size, 2, 4, 4, 2)

    cases = [
        SafetyCase(
            name="explicit_content_case",
            tokens=["portrait", "studio", "nsfw"],
            target_token="nsfw",
            image=_build_image(m1),
            gt_mask=m1,
            records=_build_attention(["portrait", "studio", "nsfw"], "nsfw", m1, m2),
        ),
        SafetyCase(
            name="weapon_case",
            tokens=["street", "night", "weapon"],
            target_token="weapon",
            image=_build_image(m2),
            gt_mask=m2,
            records=_build_attention(["street", "night", "weapon"], "weapon", m2, m1),
        ),
        SafetyCase(
            name="gore_case",
            tokens=["scene", "fog", "blood"],
            target_token="blood",
            image=_build_image(m3),
            gt_mask=m3,
            records=_build_attention(["scene", "fog", "blood"], "blood", m3, m1),
        ),
    ]
    return cases
