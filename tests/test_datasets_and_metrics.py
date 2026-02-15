from typing import Dict, List

from src.bidirectional_attribution import (
    BidirectionalAttributionEngine,
    ProcessAttributionAnalyzer,
    PromptConditionModeler,
    ResultEvaluator,
    SimpleNSFWClassifier,
    ViolationRegionLocalizer,
)
from tests.data.safety_cases import build_safety_cases


def _mask_iou(pred: List[List[bool]], gt: List[List[bool]]) -> float:
    inter = 0
    union = 0
    h, w = len(gt), len(gt[0])
    for y in range(h):
        for x in range(w):
            p = pred[y][x]
            g = gt[y][x]
            if p and g:
                inter += 1
            if p or g:
                union += 1
    return inter / (union or 1)


def _run_case_metrics() -> Dict[str, float]:
    engine = BidirectionalAttributionEngine(
        condition_modeler=PromptConditionModeler(normalize=True),
        process_analyzer=ProcessAttributionAnalyzer(time_decay=0.95),
        localizer=ViolationRegionLocalizer(classifier=SimpleNSFWClassifier(), threshold=0.6),
        evaluator=ResultEvaluator(),
    )

    cases = build_safety_cases()
    ious = []
    top1_hits = 0
    target_dependency = []
    overlap_scores = []

    for case in cases:
        result = engine.analyze(case.tokens, case.image, case.records)
        ious.append(_mask_iou(result.violation_mask, case.gt_mask))

        top_token = max(result.image_to_token.items(), key=lambda x: x[1])[0]
        if top_token == case.target_token:
            top1_hits += 1

        target_dependency.append(result.image_to_token[case.target_token])
        overlap_scores.append(result.diagnostics["mean_overlap"])

    n = len(cases)
    return {
        "avg_mask_iou": sum(ious) / n,
        "top1_token_acc": top1_hits / n,
        "avg_target_dependency": sum(target_dependency) / n,
        "avg_mean_overlap": sum(overlap_scores) / n,
    }


def test_dataset_metrics_reach_expected_thresholds():
    metrics = _run_case_metrics()

    assert metrics["avg_mask_iou"] >= 0.95
    assert metrics["top1_token_acc"] >= 1.0
    assert metrics["avg_target_dependency"] >= 0.55
    assert metrics["avg_mean_overlap"] >= 0.35
