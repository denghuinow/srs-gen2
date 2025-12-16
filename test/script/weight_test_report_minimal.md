# 权重调整测试报告 - minimal_en_iter10

## 测试目标

通过调整权重，使加权平均分按以下顺序递增：
1. r_base
2. d_base
3. no-explore-clarify
4. no-clarify
5. iter1
6. iter2
7. iter3
8. iter4
9. iter5
10. iter6
11. iter7
12. iter8
13. iter9
14. iter10

## 测试结果总结

### 当前权重配置

```
FUNCTIONAL: 0.25
BUSINESS_FLOW: 0.15
BOUNDARY: 0.10
EXCEPTION: 0.20
DATA_STATE: 0.20
CONSISTENCY_RULE: 0.10
```

**结果**: 满足 10/13 个相邻对

**问题**:
- no-explore-clarify (44.59) >= no-clarify (43.22)
- iter5 (56.48) >= iter6 (55.86)
- iter8 (57.69) >= iter9 (56.07)

### 最佳配置（通过自动搜索找到）

```
FUNCTIONAL: 0.4511
BUSINESS_FLOW: 0.1191
BOUNDARY: 0.0756
EXCEPTION: 0.1421
DATA_STATE: 0.0656
CONSISTENCY_RULE: 0.1464
```

**结果**: 满足 10/13 个相邻对

**问题**:
- no-explore-clarify (49.36) >= no-clarify (49.29)
- iter5 (63.79) >= iter6 (62.91)
- iter8 (64.85) >= iter9 (63.74)

## 问题分析

### 1. no-explore-clarify >= no-clarify

**维度差异**:
- no-clarify在FUNCTIONAL上略高（+0.89）
- no-explore-clarify在BUSINESS_FLOW、BOUNDARY、DATA_STATE上更高

**原因**: no-clarify在FUNCTIONAL上的优势不足以抵消no-explore-clarify在其他维度上的优势。

### 2. iter5 >= iter6

**维度差异**:
- iter6在BUSINESS_FLOW上更高（+1.58）
- iter5在FUNCTIONAL和EXCEPTION上更高（-0.74和-1.53）

**原因**: iter5在FUNCTIONAL和EXCEPTION上的优势超过了iter6在BUSINESS_FLOW上的优势。

### 3. iter8 >= iter9

**维度差异**:
- iter9在FUNCTIONAL和EXCEPTION上更高（+0.94和+1.96）
- iter8在BOUNDARY、DATA_STATE、CONSISTENCY_RULE上更高（-2.40、-2.93、-5.92）

**原因**: iter8在CONSISTENCY_RULE上的巨大优势（-5.92）超过了iter9在FUNCTIONAL和EXCEPTION上的优势。

## 结论

经过大量测试（包括随机搜索、智能搜索、精细调整等），**无法找到完全满足目标顺序的权重配置**。

主要原因：
1. **阶段间差异很小**: 某些相邻阶段（如iter5和iter6，iter8和iter9）之间的维度得分差异非常小，且存在反向差异。
2. **维度冲突**: 某些阶段在部分维度上更高，但在其他维度上更低，导致无论怎么调整权重，都无法完全满足递增顺序。
3. **数据特性**: 评估数据本身可能就存在这些阶段在某些维度上的反向差异，这是数据本身的特性，而非权重配置的问题。

## 建议

1. **接受部分不满足**: 当前最佳配置满足10/13个相邻对，可以考虑接受这个结果。
2. **调整目标顺序**: 如果某些阶段的顺序不是必须的，可以考虑调整目标顺序。
3. **重新评估数据**: 检查评估数据本身是否存在问题，某些阶段的评估结果可能需要重新审视。
4. **使用其他指标**: 除了加权平均分外，可以考虑使用其他评估指标来排序。

## 推荐权重配置

基于测试结果，推荐使用以下权重配置（满足10/13个相邻对）：

```python
weights = {
    "FUNCTIONAL": 0.4511,
    "BUSINESS_FLOW": 0.1191,
    "BOUNDARY": 0.0756,
    "EXCEPTION": 0.1421,
    "DATA_STATE": 0.0656,
    "CONSISTENCY_RULE": 0.1464,
}
```

该配置相比当前权重，主要变化：
- 大幅增加FUNCTIONAL权重（0.25 → 0.4511）
- 降低BUSINESS_FLOW权重（0.15 → 0.1191）
- 降低BOUNDARY权重（0.10 → 0.0756）
- 降低EXCEPTION权重（0.20 → 0.1421）
- 大幅降低DATA_STATE权重（0.20 → 0.0656）
- 增加CONSISTENCY_RULE权重（0.10 → 0.1464）


