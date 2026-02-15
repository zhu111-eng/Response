import random

from src.bidirectional_attribution import (
    AttentionRecord,
    BidirectionalAttributionEngine,
    ProcessAttributionAnalyzer,
    PromptConditionModeler,
    ResultEvaluator,
    SimpleNSFWClassifier,
    ViolationRegionLocalizer,
)


def _rand_attention(layers, heads, tokens, h, w, seed=7):
    rnd = random.Random(seed)
    return [
        [
            [
                [[rnd.random() for _ in range(w)] for _ in range(h)]
                for _ in range(tokens)
            ]
            for _ in range(heads)
        ]
        for _ in range(layers)
    ]


def test_bidirectional_pipeline_runs_and_normalizes():
    tokens = ["girl", "beach", "nsfw"]
    records = [
        AttentionRecord(10, _rand_attention(2, 4, len(tokens), 8, 8, seed=1)),
        AttentionRecord(20, _rand_attention(2, 4, len(tokens), 8, 8, seed=2)),
        AttentionRecord(30, _rand_attention(2, 4, len(tokens), 8, 8, seed=3)),
    ]

    image = [[[1.0, 0.0, 0.0] for _ in range(8)] for _ in range(8)]

    engine = BidirectionalAttributionEngine(
        condition_modeler=PromptConditionModeler(normalize=True),
        process_analyzer=ProcessAttributionAnalyzer(time_decay=0.95),
        localizer=ViolationRegionLocalizer(classifier=SimpleNSFWClassifier(), threshold=0.5),
        evaluator=ResultEvaluator(),
    )

    result = engine.analyze(tokens=tokens, image=image, attn_records=records)

    assert set(result.token_to_image.keys()) == set(tokens)
    assert len(result.violation_mask) == 8 and len(result.violation_mask[0]) == 8
    assert 0.99 <= sum(result.image_to_token.values()) <= 1.01
    assert "mean_overlap" in result.diagnostics
