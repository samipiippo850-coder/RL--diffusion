# 离散策略测度原型

该目录实现有限动作 Gibbs 更新、复制子向量场、固定参考策略的 KL 正则化折扣 MDP、条件残差汇总。用于检验数学与接口，不是扩散模型、自走棋 Agent 或 Schrödinger Bridge。

在 `path_measure_preparation` 目录运行（依赖 NumPy）：

```bash
python -m agent_measure.demo
python -m unittest test_agent_measure -v
```

`policy.gibbs_policy` 屏蔽非法动作并严格保持参考分布支持；参考概率为零的动作不会自行恢复。`replicator_field` 的零向量不证明最优，尤其是概率为零的动作可能有更高价值；`best_legal_advantage` 用于提示该现象。正则化 Gibbs 固定点一般不会让原始复制子向量场为零，因为 KL 正则化的目标不同；不要将两者的残差混为一谈。

`mdp.solve_kl_mdp` 是已知有限转移、固定对手、`gamma<1`、正参考概率的玩具验证，用确定的 Bellman 残差检查结果。有限回合自走棋需显式编码时间和终止，不能直接套用该玩具模型的收敛结论。`residual_summary` 只统计结果与预测条件均值的差；对真实数据应使用决策时可见信息，基于独立或交叉拟合数据预测，按信息分组检查偏差，并考虑相关性和对手变化。分组均值接近零不是鞅差证明。

`final_rank_reward(None)` 保留未知/截断结果；它不计算名次。模拟器的平局、回合上限和截断语义仍需先验证。下一步接入 TFT 环境时需要分别实现动作前观测采集、mask、可靠排名与固定对手数据管道；不能直接对动作编号做高斯扩散。
