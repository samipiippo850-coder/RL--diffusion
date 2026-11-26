# Diffuser 复现记录

- 上游仓库：https://github.com/jannerm/diffuser ，本地固定提交 `7ea422860cc0106e5ca5949d980f04b799d5462c`；服务器解包目录 `/root/autodl-tmp/path_measure_preparation/diffuser_7ea4228`。这是当前公开提交，不证明与论文训练时提交完全一致。
- 论文表 2 的 Hopper-medium 为 74.3 ± 1.4，误差为 150 次规划评估的标准误。作者仓库报告 74.3 ± 1.36。目录编号 0–149 不足以证明是随机种子值。
- 该提交 `config/locomotion.py` 对 diffusion、values、plan 均设 `seed=None`；`diffuser/utils/setup.py` 在 seed 为 None 时不设置随机状态。公开文档未给出这 150 次的实际 seed 序列，不应宣称精确还原。
- 显式传入 `--seed` 时，`utils/setup.py` 设置 Python、NumPy 和 PyTorch 随机源；`plan_guided.py` 将其传给数据集，`datasets/sequence.py` 调用 `env.seed(seed)`。仍需在实际安装版本中验证可重复性。
- 基线应先按作者预训练 diffusion 和 value 模型、对应配置及 Hopper-medium-v2 数据、原版引导评估进行；获得可运行结果后采用显式记录的 150 个评估种子，记录环境、PyTorch、NumPy、Python 随机源。需检查环境 seed 是否得到设置，固定数值不能保证跨硬件完全确定。
- 论文附录描述的训练参数与仓库当前配置存在差异（如学习率、步数）；报告时分别标识论文版本和代码提交，勿将现行代码默认值称为原始训练配置。原始环境列于上游 environment.yml，包括 Python 3.8、Gym 0.18、MuJoCo-py 2.0.2.13 和固定 D4RL commit。
- 优化实验须基线完成后进行：基线与 Gibbs 训练从相同初始化和数据开始，固定网络、训练预算、价值引导模型和配对评估种子。beta=0 对照与按样本 loss 加权正确性先验证。训练随机种子与规划评估种子分开报告；不把 150 次评估当成 150 次独立训练。
- 尚未完成环境、检查点、Hopper 数据获取和实际基线评估；二维合成实验不是论文复现。
- 服务器隔离环境 `/root/autodl-tmp/path_measure_preparation/env38` 已创建 Python 3.8.20、NumPy 1.23.5；pip 24.0、setuptools 65.7.0、wheel 0.38.4 已安装。RTX 4090 不支持原 CUDA 11.1 构建，计划评估 CUDA 11.8 版本 PyTorch 2.0.1 并记录偏差；下载尝试卡住，尚未安装。Gym、mujoco-py、D4RL 和检查点尚未安装或验证。
- 后续已在本地下载 PyTorch 2.0.1+cu118 安装包、论文指定 D4RL 提交归档和 MuJoCo 2.0 等依赖；服务器上传期间连接中断，SSH 新连接被关闭。服务器上可能留有不完整文件，恢复后须检查大小和哈希，禁止直接安装部分上传的 wheel。Python 环境和文件最终状态需重新确认。
- 当前进展：PyTorch wheel 的 SHA-256 已核验并安装；Gym 0.18.0、mujoco-py 2.0.2.13、D4RL 指定提交和 mjrl（提交 3871d93763d3b49c4741e6daeaebbc605fe140dc）已安装。Hopper-medium-v2 的 reset 和一步 step 成功；Diffuser 导入与 TemporalUnet CPU 前向、反向通过。启动环境：`source /root/autodl-tmp/path_measure_preparation/activate_diffuser.sh`。
- 当前容器没有 `/dev/nvidia*`，预装及隔离 PyTorch 均报告 CUDA 不可用，GPU 回归测试待恢复设备后进行。`pip check` 仍提示 triton、dm-control、pybullet 未安装；后两者为 D4RL 其他环境依赖，triton 尚未验证，不能称全依赖完整。数据集下载、训练入口、检查点和 150 次评估尚未验证。
