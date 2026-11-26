# 路径测度优化：首轮准备

本目录独立于现有 DistDF 工程。当前交付为有限样本 Gibbs 权重诊断原型和实验设计，不包含已训练的 Diffuser、D4RL 数据或 Bridge 实现。示例数据是合成数据，不能作为性能结果。

## 运行

只需 Python 3：

```bash
python path_measure_preparation/gibbs.py path_measure_preparation/example_costs.csv --beta 0 0.1 0.5 1
python -m unittest discover -s path_measure_preparation -p 'test_*.py' -v
```

输入 CSV 每行一个固定训练样本的成本，默认列名 `cost`；结果输出样本数、有效样本量 ESS、ESS 比例、最大归一化权重、加权及均匀平均成本、相对均匀经验测度的 KL。代码按最小成本平移后指数化，避免常见溢出。成本单位会改变 beta 的含义；对比前必须固定成本定义和尺度。beta=0 是一致性对照。成本必须有限且 beta 非负。

## 接入和实验顺序

1. 在 Ubuntu 独立环境中获取 CleanDiffuser，记录确切 commit、依赖、CUDA、MuJoCo、数据集和许可证；先核实选定分支是否确实包含并能运行目标 pipeline。lightning 分支 README 明确提示部分 pipeline 尚未迁移或未充分测试，不能假定文档所列路径可直接运行；必要时评估 main 分支并记录偏差。不要混装到现有 DistDF 环境。
2. 完成 Hopper 数据加载、环境 reset/step 和小批量训练与评估 smoke test，再固定数据切分、轨迹窗口、终止/截断处理、归一化统计、网络、去噪及规划设置。记录首步耗时和显存，随后再估算预算。
3. 明确采样单位：完整 episode 与固定长度训练窗口不同。若训练窗口均匀采样，必须为每个窗口定义一致的成本和对应经验参考分布；避免把 episode 权重误用为窗口权重，特别注意不同 episode 长度造成的隐式重复。先使用窗口内真实奖励之和的负值，记录折扣、padding mask 和终止边界；约束成本仅在有明确可测定义时加入。
4. 先完成 A 组原始训练/规划；B 组复用同一骨架。计算每个样本的去噪误差（先对有效维度求均值，再跨样本加权）。若训练样本均匀抽取，可用训练集预先计算并固定的全局归一化权重，或使用无放回/有放回的明确重采样方案。批内自归一化是有偏近似，batch size 1 时完全失效；需单独标注。不要同时按权重重采样又加权 loss，以免重复加权。权重裁剪会改变目标分布，必须作为独立消融记录。
5. 先在训练集扫描 beta 和全局 ESS；预先选定阈值和 beta，使用验证集选择设置，测试集和环境评估仅用于最终报告。记录归一化收益、原始回报、成功/终止、随机种子、训练步数、采样步数与实际 GPU 时间。固定评估初始状态和每种子 episode 数，报告各训练种子的均值及不确定性。若使用价值引导，A/B 共享同一价值模型、训练数据及引导预算，并另行报告无引导结果。
6. C/D 先在二维玩具问题单独验证边缘分布和势函数，再考虑轨迹空间接入；它们不是当前原型已实现的功能。游戏时间、扩散时间、bridge 时间分开记录。

## 指标定义待冻结

ESS = 1 / sum(p_i²)，其中 p 是训练集权重归一化后的离散概率；ESS/N 随样本集合变化，应报告 N。MMD/Sinkhorn 的核、正则化、特征尺度和样本量需预先固定；coverage 需给出轨迹特征和距离阈值，连续空间熵的估计方法和维度需明确。权重集中会自然改变与原数据的距离，距离变大本身不等同生成质量下降。归一化收益只与相同环境版本及评估协议下的结果比较，文档引用的 74.3±1.36 不能直接作为本实现结果。

## 数学边界

有限经验参考测度下 Gibbs 仅给已有样本重新分配质量；生成模型可能插值或外推，但这不等于有可靠环境支持。轨迹随机变量空间的 KL 重加权与扩散/bridge 随机过程的路径空间 KL 是不同对象，不能直接等同。Girsanov 控制能量公式还需要初始分布一致、控制可积及适当绝对连续性等条件；初始分布变化时一般还需初始 KL 项。

上游参考：[CleanDiffuser lightning](https://github.com/CleanDiffuserTeam/CleanDiffuser/tree/lightning)、[Diffuser](https://github.com/jannerm/diffuser)、[TSBM](https://github.com/maxencenoble/twisted-sb-matching)、[TSBM 论文](https://arxiv.org/abs/2607.16987)。
