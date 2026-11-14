# LGCS 可运行研究 demo

对应当前《Lesion-Graph Representations and Context Selection for Label-Efficient Multi-Disease Fundus Screening》第三章。这个工程实际执行源域小模型训练、冻结图表示、32+8列接口、按病种平衡选择、上下文预测与离线核验。默认 CPU，不下载眼底数据或预训练权重。

默认图编码器是小型演示网络，默认预测器是明确标识的 RBF 上下文替代器，**不是 SIGraph/TabICL 的复现成绩**。精确实现部分和正式版替换入口见 METHOD_MAP.md。所有运行指标来自随包合成样本，只用于检查流程。

## 运行

Python 3.12，建议独立环境。CPU 安装示例（[PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)）：

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5
python run_demo.py
python -m unittest discover -s tests -v
```

入口基于自身目录定位配置和数据，从其他工作目录启动也可；`--output` 指定输出目录，`--config` 指定配置文件。默认单线程 CPU、固定种子。

## 随包内容

- `data/sample.npz`：96幅源训练、80幅候选库、24幅查询的32×32合成视网膜样式图，及 FOV、唯一标识、合成等级和两个二值属性。
- `data/generate_data.py`：同一种子可重新生成数值相同的数据。`metadata.json` 与 DATA_CARD.md 记录来源、单位和限制。
- `src/core.py`：可复用图编码器、统计量、源域PCA接口、选择器、上下文预测器和分数并列处理的AP计算。
- `outputs/demo`：本机实际运行生成的模型、接口参数、预测、每个查询的已用标签与所选上下文，以及运行诊断。

`metrics.json` 中的 `source_model_unchanged_in_target` 和 `full_library_max_probability_delta` 分别检查冻结权重与全库端点一致性。各规则的合成 AP 原样保留，无论升降；不可用于填写正文实测结果。图编码器空图仍参与预测，无效FOV则显式失败，不转成阴性。

## 接入真实数据

按 DATA_CARD.md 中的数组模式准备有许可的图像、FOV与患者标识，先按患者划分，再建立源域训练、目标支持池、固定查询。源域只提供DR等级；目标各病种标签独立揭示，未知不能填0。正式40列接口需要源域数值秩至少32。

将 `TinySourceGraph` 替换为经过审计的官方架构，导出融合前64维示例接口对应的实际 `C_t` 读出和累计分配；接口不假定正式读出必须64维。将 `ContextPredictor` 的 `estimator_factory` 传入已明确加载官方 TabICL v1 检查点的构造函数。每个支持上下文创建新分类器，仅在所选支持行拟合预处理；核验正类列、单条/批量一致性与全库端点。完整的官方模型资源、30次支持抽样、病种字典审计与置信区间实验尚未执行。

