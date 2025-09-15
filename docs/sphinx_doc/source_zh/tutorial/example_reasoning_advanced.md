# Off-Policy RFT


让我们继续使用 [之前的 GSM8k 例子](./example_reasoning_basic.md)，并展示 Trinity-RFT 提供的一些高级功能，即 off-policy 或异步 RFT 模式。

(OPMD)=
## OPMD：一种原生的 off-policy 强化学习算法

作为 Trinity-RFT 的一个实验性功能，我们开发了一种极其简单的 off-policy 强化学习算法，称为 OPMD（Online Policy Mirror Descent，灵感来自 [Kimi k1.5](https://arxiv.org/abs/2501.12599)）。
该算法的设计与分析详见[Trinity-RFT 技术报告](https://arxiv.org/abs/2505.17826)的附录A。
本例子对应的配置文件为 [`opmd_gsm8k.yaml`](https://github.com/modelscope/Trinity-RFT/blob/main/examples/opmd_gsm8k/opmd_gsm8k.yaml)。

要尝试 OPMD 算法，请运行：
```shell
trinity run --config examples/opmd_gsm8k/opmd_gsm8k.yaml
```

注意，在此配置文件中，`sync_interval` 被设置为 10，也就是说，explorer 和 trainer 每 10 个训练步骤才同步一次模型权重，这导致了一个具有挑战性的 off-policy 场景（在 RFT 过程中可能出现剧烈的分布偏移）。

下图中的红色曲线展示了 OPMD 学习过程的一个示例。
由于 explorer 的模型权重在前 10 步保持不变，其得分也保持平稳。
然后，在第 10 步结束时，explorer 和 trainer 完成模型权重同步，我们在第 11 步观察到得分突然上升，这表明前 10 步的 off-policy 学习是有效的。
类似的性能提升在第 21 步再次出现，最终收敛的得分与在准在线策略情况下（`sync_interval=2`）GRPO 所达到的结果相当。

![opmd](../../assets/opmd-curve.png)
