# 双向归因分析框架（文生图违规生成）

本实现对应一个可扩展的研究原型，围绕扩散模型跨注意力构建四模块流水线：

1. **提示词条件建模模块**：在采样期间捕获并归一化跨注意力。
2. **生成过程归因分析模块**：沿时间步、注意力头、网络层进行加权聚合。
3. **违规区域定位模块**：通过 NSFW 分类器热图阈值化得到风险区域。
4. **结果分析与评估模块**：输出重叠度与风险强度等指标。

并进一步提供双向归因：
- 词到图像：`token_to_image`
- 图像到词：`image_to_token`

## 快速使用

```python
from random import Random
from src import (
    AttentionRecord, PromptConditionModeler, ProcessAttributionAnalyzer,
    ViolationRegionLocalizer, ResultEvaluator,
    BidirectionalAttributionEngine, SimpleNSFWClassifier,
)

rnd = Random(0)

def rand_attn(layers, heads, tokens, h, w):
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

records = [
    AttentionRecord(timestep=10, attention=rand_attn(2, 4, 3, 16, 16)),
    AttentionRecord(timestep=20, attention=rand_attn(2, 4, 3, 16, 16)),
]

engine = BidirectionalAttributionEngine(
    condition_modeler=PromptConditionModeler(),
    process_analyzer=ProcessAttributionAnalyzer(),
    localizer=ViolationRegionLocalizer(SimpleNSFWClassifier(), threshold=0.6),
    evaluator=ResultEvaluator(),
)

image = [[[1.0, 0.2, 0.2] for _ in range(16)] for _ in range(16)]
result = engine.analyze(["person", "bedroom", "nsfw"], image, records)
print(result.image_to_token)
```

## 测试数据集与结果指标

新增了 3 组小型安全测试集（位于 `tests/data/safety_cases.py`）：

- `explicit_content_case`
- `weapon_case`
- `gore_case`

每组数据都包含：
- 提示词 token 列表
- 目标违规 token（期望被反向归因识别）
- 合成图像
- 违规区域 GT 掩码
- 多时间步跨注意力记录

对应测试 `tests/test_datasets_and_metrics.py` 会计算以下指标：

- `avg_mask_iou`：预测违规区域与 GT 掩码的平均 IoU
- `top1_token_acc`：图像→词反向归因 top1 命中率
- `avg_target_dependency`：目标违规 token 的平均依赖分数
- `avg_mean_overlap`：模型输出 `mean_overlap` 的数据集平均值

运行方式：

```bash
pytest -q
```
