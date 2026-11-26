# 离散扩散策略与自走棋智能体

这是一个研究原型：在开源 TFT Set 4 风格模拟器中，探索合法动作掩码下的离散扩散、模仿学习、组相对策略更新与局面分支分析。当前结果为小规模实验，**尚未证明扩散模块、残差引导或 GRPO 稳定优于脚本**。

项目从路径测度优化研究转向策略游戏智能体；历史文档保留了早期计划。最新实现和结果请以本文件及 [研究进展](docs/PROGRESS.md) 为准。

## 当前内容

- [agent_measure](path_measure_preparation/agent_measure)：单动作 categorical 扩散、分层合法动作策略、条件价值估计、去噪轨迹 GRPO 和残差工具。
- [经济脚本模仿](path_measure_preparation/tft_econ_imitation.py)：数据采集、逐局清洗、分层行为克隆和分层扩散。
- [DAgger 试验](path_measure_preparation/tft_dagger_experiment.py)：模型状态上的脚本标注，并与等量新增脚本数据比较。
- [GRPO 试验](path_measure_preparation/tft_grpo_experiment.py)：真实排名与示范偏好奖励的 GRPO 风格小试验。
- [局面分支试验](path_measure_preparation/tft_segment_pilot.py)：在复制局面上强制一个动作，再用经济脚本继续对局。
- [逐局结果](path_measure_preparation/training_results_20261124)：实验指标和探索结果；解释见研究进展。
- [模拟器补丁](patches/tft-simulator-item-fixes.patch)：针对固定版本上游模拟器的本地修改；上游许可证见 [third_party](third_party/TFTMuZeroAgent-LICENSE)。

## 最近结果

经济脚本模仿实验使用 64 局训练示范，并按游戏种子留出验证和测试。16 场实际测试中，分层扩散平均排名 2.63，经济脚本 3.25；样本量小，尚不能认定稳定超越。之后 24 种子 DAgger 对照中，原分层扩散、DAgger 和等量新增脚本数据的平均排名分别为 2.67、3.13 和 3.42；差异尚不可靠。

局面分析初试找到 5 个低血量与 8 个升级/刷新局面，104 条分支自然结束。每个局面的 4 对后续结果只能用于提出候选，不能证明通用策略优势。两个事后选中局面的新增随机种子复查记录尚未完成，见 [研究进展](docs/PROGRESS.md)。

## 环境与复现范围

实际运行环境为 Linux、Python 3.12、PyTorch 2.8.0+cu128；模拟器源自 [TFTMuZeroAgent](https://github.com/silverlight6/TFTMuZeroAgent)，固定提交 `a197a2a5bfab26c2900ac77e7b48c349efc14016`。本仓库不包含上游完整源码、虚拟环境、原始对局数据或连接凭据。上游使用 Apache-2.0，相关许可证原文已保留。

目前若干实验脚本仍使用固定路径 `/root/autodl-tmp/path_measure_preparation`，并依赖前序实验的归一化检查点、示范数据或模型文件。**不能仅克隆本仓库就从头运行所有历史实验。** 安装与文件依赖、哪些检查点已包含及本地补丁应用方法见 [复现说明](docs/REPRODUCIBILITY.md)。运行前请使用新输出目录，历史脚本部分会覆盖同名结果。

## 研究边界

当前扩散只生成**单个离散动作**，不是轨迹扩散或 Schrödinger Bridge。价值预测残差不自动构成鞅差；局部概率倾斜不等于已学习连续 SDE 向量场，也没有证明不动点。偏好奖励依据模拟器名次学习，不是人类专家奖励，也不能证明恢复了真实奖励函数。动作掩码合法不保证环境实际执行成功。
