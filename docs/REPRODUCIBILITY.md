# 环境与复现说明

这些步骤整理自已有运行环境；尚未在全新机器上完整复验。上游模拟器独立许可为 Apache-2.0，本仓库只提供针对固定提交的差异补丁，完整源码需从上游获取。

## 1. 代码与模拟器

当前实验脚本多数固定使用 `/root/autodl-tmp/path_measure_preparation` 作为 `BASE`；以下示例保留这个路径约定。在新机器使用其他路径前，应先检查脚本中的绝对路径。

```bash
# 假设本仓库位于 $REPO，Python 3.12 已安装。
export REPO="$PWD"
export B=/root/autodl-tmp/path_measure_preparation
mkdir -p "$B"
cp -a "$REPO/path_measure_preparation/." "$B/"
git clone https://github.com/silverlight6/TFTMuZeroAgent.git "$B/tft_item_fixed_v2"
git -C "$B/tft_item_fixed_v2" checkout a197a2a5bfab26c2900ac77e7b48c349efc14016
git -C "$B/tft_item_fixed_v2" apply --check "$REPO/patches/tft-simulator-item-fixes.patch"
git -C "$B/tft_item_fixed_v2" apply "$REPO/patches/tft-simulator-item-fixes.patch"
python3.12 -m venv "$B/tft_env"
"$B/tft_env/bin/python" -m pip install -e "$B/tft_item_fixed_v2" pytest torch
export PYTHONPATH="$B/tft_item_fixed_v2:$B"
```

运行时曾使用 Python 3.12.3 和 PyTorch 2.8.0+cu128；`pip install torch` 的 CUDA 版本取决于安装源和机器配置。上游 `requirements.lock` 记录了当时的 Python 包版本，跨平台兼容性未验证。示例不会自动安装 CUDA 驱动。

在新实例中核对到现有环境的 NumPy 2.3.2、Gymnasium 1.4.0 和 PettingZoo 1.27.0；摘要见 [requirements-observed.txt](../requirements-observed.txt)。它是环境观察记录，不是经过全新环境验证的完整锁文件。新实例检查时 `torch.cuda.is_available()` 为 `False`，与先前用于训练的 GPU 实例不同。

补丁包括装备返还容量检查、装备溢出队列、队列更新以及悠米施法死亡检查。溢出队列未编码进原始固定长度观测，是实验性近似；相关日志检测也不保证能发现所有环境状态错误。请勿在不固定版本的上游目录盲目应用补丁。

## 2. 单元测试

在上面环境下，从 `B` 目录运行：

```bash
cd "$B"
"$B/tft_env/bin/python" -m unittest test_gibbs test_agent_measure test_discrete_diffusion test_conditional_value test_diffusion_grpo test_hierarchical_policy test_econ_imitation test_dagger_experiment test_segment_pilot test_item_safety test_yuumi_cast -v
```

这些测试检查数学关系、动作合法掩码和数据清洗接口，不代表完整对局或模型性能已经复现。源仓库早期 `path_measure_preparation/README.md` 保留初始计划，其中包含已过时的项目阶段说明；以本目录和研究进展为准。

整理仓库时，以上 31 项测试在现有 Windows 本地虚拟环境和已有补丁后的模拟器目录中通过；补丁可逆向检查通过，四个修改文件按统一换行后的内容哈希与服务器一致。这不是全新环境安装验证。

## 3. 检查点与原始数据

训练与评估程序会读取 `B` 下的前序文件，例如：

- `tft_pilot_20261121/policy.pt`：旧观测均值、标准差及基线模型；当前 `tft_intense_game.py` 在进入对局时会读取它。
- `tft_econ_imitation_20261124/hier_diffusion.pt`：经济脚本模仿模型。
- `tft_dagger_20261124/dagger.pt` 与 `fresh_teacher.pt`：DAgger 及等量脚本示范对照模型。
- `tft_econ_imitation_20261124/data/`：按游戏保存的示范原始观测及掩码；DAgger 训练脚本会读取其中的原始 64 局。
- `tft_segment_pilot_20261124/metrics.json`：候选局面复查脚本的输入。

仓库 `checkpoints/` 只包括已成功传输并校验或本地验证的文件；未包含的检查点和原始示范数据需要从原实验数据盘取得或重新采集。若文件存在，可将 `checkpoints/` 下对应的子目录复制到 `B`，保持上述相对路径。只有聚合指标 JSON 的实验不能由此完整恢复训练数据。

已收录 `tft_pilot_20261121/policy.pt`、`tft_econ_imitation_20261124/hier_diffusion.pt` 和 `hier_bc.pt`；前两者与当前服务器的 SHA-256 一致，后者在本地成功加载。DAgger 的检查点是否收录以 `checkpoints/tft_dagger_20261124/` 的实际内容为准。原始示范数据未收录。

```bash
cp -a "$REPO/checkpoints/." "$B/"
```

**注意：** `tft_econ_imitation.py`、`tft_dagger_experiment.py` 等脚本使用固定输出目录；直接重跑可能覆盖同名文件。修改输出路径或在全新目录使用，并记录代码、数据切分、随机种子和输出位置。

## 4. 现有数据的统计限制

所有排名来自该模拟器；同回合多名玩家淘汰时的内部顺序可能带来座位影响。相同初始随机种子下，策略选择不同动作后会消耗不同随机流，不能当作完全相同的后续对局。

候选局面比较的干预仅为**强制当前一个动作、随后统一使用经济脚本**。后续随机种子作配对并不保证分叉后的随机流逐项对齐；模拟结果仅针对所选局面和该后续策略。两个示例快照复制重放一致，只是有限可行性验证。
