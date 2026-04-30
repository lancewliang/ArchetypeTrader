"""VQ encoder-decoder 模型组件包。

包含: encoder 输入适配、向量量化、loss、整体模型组装。

关键约束:
- decoder 必须为单向因果结构；禁止 bidirectional / 全 horizon pooling / 未来 state pooling。
- VectorQuantizer 默认 ``kmeans_warmup + ema``；严格复现论文公式 (4) 时切换为 ``random_normal + gradient``。
- 所有归一化（reward）只能在 train demonstration 上拟合；val/test 仅 transform。
"""
