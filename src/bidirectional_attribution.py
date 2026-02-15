from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


# 注意力张量结构: [layers][heads][tokens][h][w]
AttentionTensor = List[List[List[List[List[float]]]]]
Map2D = List[List[float]]
Mask2D = List[List[bool]]


@dataclass
class AttentionRecord:
    timestep: int
    attention: AttentionTensor


@dataclass
class AttributionResult:
    tokens: List[str]
    violation_mask: Mask2D
    token_to_image: Dict[str, Map2D]
    image_to_token: Dict[str, float]
    diagnostics: Dict[str, float]


class NSFWClassifier(Protocol):
    def predict_heatmap(self, image: List[List[List[float]]]) -> Map2D:
        ...


def _zeros(h: int, w: int) -> Map2D:
    return [[0.0 for _ in range(w)] for _ in range(h)]


def _shape_5d(x: AttentionTensor) -> Tuple[int, int, int, int, int]:
    return len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]), len(x[0][0][0][0])


@dataclass
class PromptConditionModeler:
    normalize: bool = True

    def capture(self, records: Sequence[AttentionRecord]) -> List[AttentionRecord]:
        captured: List[AttentionRecord] = []
        for r in records:
            attn = r.attention
            if self.normalize:
                l, h, t, hh, ww = _shape_5d(attn)
                normed: AttentionTensor = []
                for li in range(l):
                    heads = []
                    for hi in range(h):
                        tokens = []
                        for ti in range(t):
                            amap = attn[li][hi][ti]
                            s = sum(sum(row) for row in amap) or 1e-8
                            tokens.append([[v / s for v in row] for row in amap])
                        heads.append(tokens)
                    normed.append(heads)
                captured.append(AttentionRecord(r.timestep, normed))
            else:
                captured.append(r)
        return captured


@dataclass
class ProcessAttributionAnalyzer:
    layer_weights: Optional[List[float]] = None
    head_weights: Optional[List[float]] = None
    time_decay: float = 0.98

    def aggregate(self, records: Sequence[AttentionRecord]) -> List[Map2D]:
        if not records:
            raise ValueError("records 不能为空")

        records = sorted(records, key=lambda x: x.timestep)
        l, h, t, hh, ww = _shape_5d(records[0].attention)

        layer_w = self.layer_weights[:] if self.layer_weights else [1.0] * l
        head_w = self.head_weights[:] if self.head_weights else [1.0] * h
        ls, hs = sum(layer_w) or 1e-8, sum(head_w) or 1e-8
        layer_w = [x / ls for x in layer_w]
        head_w = [x / hs for x in head_w]

        fused = [_zeros(hh, ww) for _ in range(t)]

        n = len(records)
        for idx, r in enumerate(records):
            tw = self.time_decay ** (n - 1 - idx)
            for ti in range(t):
                for yi in range(hh):
                    for xi in range(ww):
                        val = 0.0
                        for li in range(l):
                            for hi in range(h):
                                val += layer_w[li] * head_w[hi] * r.attention[li][hi][ti][yi][xi]
                        fused[ti][yi][xi] += tw * val

        for yi in range(hh):
            for xi in range(ww):
                s = sum(fused[ti][yi][xi] for ti in range(t)) or 1e-8
                for ti in range(t):
                    fused[ti][yi][xi] /= s

        return fused


@dataclass
class ViolationRegionLocalizer:
    classifier: NSFWClassifier
    threshold: float = 0.6

    def locate(self, image: List[List[List[float]]]) -> Tuple[Map2D, Mask2D]:
        heatmap = self.classifier.predict_heatmap(image)
        mask = [[v >= self.threshold for v in row] for row in heatmap]
        return heatmap, mask


@dataclass
class ResultEvaluator:
    def evaluate(self, token_maps: Dict[str, Map2D], violation_mask: Mask2D, heatmap: Map2D) -> Dict[str, float]:
        h, w = len(violation_mask), len(violation_mask[0])
        risk_count = sum(1 for y in range(h) for x in range(w) if violation_mask[y][x])

        if not token_maps:
            return {"mean_overlap": 0.0, "risk_area_ratio": risk_count / (h * w)}

        overlaps: List[float] = []
        for amap in token_maps.values():
            m = max(max(row) for row in amap) or 1e-8
            overlap = 0.0
            for y in range(h):
                for x in range(w):
                    if violation_mask[y][x]:
                        overlap += amap[y][x] / m
            overlaps.append(overlap / (risk_count or 1))

        risk_sum = sum(heatmap[y][x] for y in range(h) for x in range(w) if violation_mask[y][x])
        return {
            "mean_overlap": sum(overlaps) / len(overlaps),
            "risk_area_ratio": risk_count / (h * w),
            "risk_intensity": risk_sum / (risk_count or 1),
        }


class BidirectionalAttributionEngine:
    def __init__(
        self,
        condition_modeler: PromptConditionModeler,
        process_analyzer: ProcessAttributionAnalyzer,
        localizer: ViolationRegionLocalizer,
        evaluator: ResultEvaluator,
    ) -> None:
        self.condition_modeler = condition_modeler
        self.process_analyzer = process_analyzer
        self.localizer = localizer
        self.evaluator = evaluator

    def analyze(
        self,
        tokens: Sequence[str],
        image: List[List[List[float]]],
        attn_records: Sequence[AttentionRecord],
    ) -> AttributionResult:
        tokens = list(tokens)
        captured = self.condition_modeler.capture(attn_records)
        fused = self.process_analyzer.aggregate(captured)
        if len(fused) != len(tokens):
            raise ValueError("token 数量与注意力 token 维度不一致")

        heatmap, mask = self.localizer.locate(image)
        token_to_image = {tok: fused[i] for i, tok in enumerate(tokens)}
        img2tok_scores = self._image_to_token_dependency(fused, mask)
        image_to_token = {tok: img2tok_scores[i] for i, tok in enumerate(tokens)}
        diagnostics = self.evaluator.evaluate(token_to_image, mask, heatmap)

        return AttributionResult(tokens, mask, token_to_image, image_to_token, diagnostics)

    @staticmethod
    def _image_to_token_dependency(fused: List[Map2D], violation_mask: Mask2D) -> List[float]:
        t = len(fused)
        h, w = len(violation_mask), len(violation_mask[0])
        scores = [0.0] * t
        for ti in range(t):
            s = 0.0
            for y in range(h):
                for x in range(w):
                    if violation_mask[y][x]:
                        s += fused[ti][y][x]
            scores[ti] = s
        total = sum(scores) or 1e-8
        return [s / total for s in scores]


class SimpleNSFWClassifier:
    def predict_heatmap(self, image: List[List[List[float]]]) -> Map2D:
        h, w = len(image), len(image[0])
        heat = _zeros(h, w)
        for y in range(h):
            for x in range(w):
                px = image[y][x]
                r, g, b = px[0], px[1], px[2]
                if max(r, g, b) > 1.0:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                heat[y][x] = max(0.0, min(1.0, 0.7 * r + 0.3 * lum))
        return heat
