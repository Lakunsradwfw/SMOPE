# SMoPE + SplitLoRA 异构梯度保护 · 项目指南

> **角色设定（每次对话开头读取本文件即可对齐上下文）**
> - **你（用户）**：计算机专业在读科研学生，正在做持续学习（Continual Learning, CL）方向的研究。
> - **我（Reasonix Code）**：你的权威科研导师，拥有非常前沿的科研能力，负责审阅思路、指出可优化点、帮助落地实现。
> - **工作上下文**：每次对话以本文件为共享上下文。

---

## 1. 论文基线 & 参考文献

### 1.1 Baseline：SMoPE

> **论文**：*One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning*
> **代码**：https://github.com/Minhchuyentoancbn/SMOPE
> **核心思想**：将共享 prompt 组织为多个 "prompt experts" 放入稀疏 MoE 架构，每个输入只激活 top-k 相关专家，用 prompt-attention score aggregation + adaptive noise + prototype-based loss 实现高效的 prompt-based CL。

**关键引用**（来自原文）：

> *"Prompt-based methods have recently gained prominence in Continual Learning (CL) due to their strong performance and memory efficiency. A prevalent strategy in this paradigm assigns a dedicated subset of prompts to each task, which, while effective, incurs substantial computational overhead and causes memory requirements to scale linearly with the number of tasks. Conversely, approaches employing a single shared prompt across tasks offer greater efficiency but often suffer from degraded performance due to knowledge interference. To reconcile this trade-off, we propose SMoPE, a novel framework that integrates the benefits of both task-specific and shared prompt strategies."*

**SMoPE 核心架构**（来自原文）：

> *"The attention mechanism for each head is composed of both pre-trained and prompt components. The pre-trained attention matrix \(A^{\text{pre-trained}}_l\) is computed using standard self-attention. To construct the prompt attention matrix \(\tilde{A}^{\text{prompt}}_l\), we first calculate the average input representation \(\tilde{x}\), and evaluate the scores for all prompt experts. During training, frequently activated prompt experts are penalized by applying an adaptive noise to their scores, which promotes exploration of underutilized experts for new tasks while preserving essential knowledge in critical experts. A Top-K selection operator then identifies the most relevant experts based on these adjusted scores. The selected scores are row-expanded to form \(\tilde{A}^{\text{prompt}}_l\). Finally, \(\tilde{A}^{\text{prompt}}_l\) is concatenated with \(A^{\text{pre-trained}}_l\) to produce the final attention matrix, which is applied to the expert representations via a dot product, similar to the standard self-attention mechanism."*

### 1.2 参考文献：SplitLoRA

> **论文**：*SplitLoRA: Balancing Stability and Plasticity in Continual Learning Through Gradient Space Splitting*
> **代码**：https://github.com/qhmiao/SplitLoRA
> **核心思想**：对 LoRA 的梯度空间做 SVD 分解，将旧任务的 major subspace 作为稳定空间、minor subspace 作为可塑空间，新任务梯度中与 major subspace 重合的方向被削弱/剔除，从而平衡稳定性与可塑性。

**关键引用**（来自原文）：

> *"Continual Learning (CL) requires a model to learn multiple tasks in sequence while maintaining both stability—preserving knowledge from previously learned tasks, and plasticity—effectively learning new tasks. Gradient projection has emerged as an effective and popular paradigm in CL, where it partitions the gradient space of previously learned tasks into two orthogonal subspaces: a primary subspace and a minor subspace. New tasks are learned effectively within the minor subspace, thereby reducing interference with previously acquired knowledge."*

**SplitLoRA 核心贡献**（来自原文）：

> *"We theoretically model the impact of the gradient subspace size of previous tasks on stability and plasticity in orthogonal projection based continual learning in Theorem 4.2 and derive an approximate optimal minor subspace in CL. We introduce SplitLoRA, a novel PEFT framework. By projecting the minor subspace onto the LoRA dimension reduction matrix A_t via a random projection and optimizing only B_t, SplitLoRA ensures that updates remain confined to the minor subspace, thereby achieving an effective balance between stability and plasticity. Our method achieves state-of-the-art performance across multiple datasets, surpassing existing CL methods by 2%–5% on different datasets."*

---

## 2. 项目优化思路（原始构思 + 导师建议整合版）

### 2.1 总体思想：异构梯度保护（Heterogeneous Gradient Protection）

SplitLoRA 只处理了 LoRA 矩阵的**单一参数空间**，而 SMoPE 是一个**多组件系统**（router / expert&prompt / key&prototype）。核心创新在于：**不是对所有参数一刀切地投影，而是根据每个组件的功能角色设计不同的保护机制**。

### 2.2 组件一：Router / Gating —— 分布约束（非硬投影）

**原始思路**：
- 允许 router 较强适应性以适应新任务
- 保存旧任务的 router prototype distribution
- 对 router 梯度只限制在旧任务分布敏感方向上的变化

**导师建议（已采纳）**：
- ❌ 放弃模糊的「分布敏感方向」表述和 SplitLoRA 式硬投影
- ✅ **采用 KL 散度正则项**：对每个旧任务的每个类，保存其 router logits 的 prototype（均值向量），新任务训练时加 KL 散度正则，约束 router 在旧类 prototype 附近的输出不漂移太远
- ✅ **Task-level expert usage frequency** 作为轻量级约束基线

### 2.3 组件二：Prompt / Expert 参数 —— SplitLoRA 式梯度分裂

**原始思路**：
- 每完成一个任务，估计旧任务在每个 expert/prompt block 上的梯度空间
- 新任务梯度与旧 major subspace 重合的方向被削弱
- 高频 expert 强保护，低频/新 expert 弱约束

**导师建议（已采纳）**：
- ✅ **Shared subspace + expert-specific scaling**（替代 per-expert 独立 SVD）：
  - 所有 expert 共享一个全局旧任务 major subspace（一次 SVD，O(d³) 而非 O(K·d³)）
  - 每个 expert 有自己的 protection strength coefficient，由其被旧任务使用的频率决定
  - 高频 expert → 投影系数接近 1（几乎完全投影到 minor subspace）
  - 低频 expert → 投影系数接近 0（几乎不约束）
- ✅ 考虑 co-activation pattern：经常同时激活的 expert 组可做 joint gradient space 估计（扩展讨论）

### 2.4 组件三：Key / Prototype 对齐 —— 几何稳定性

**原始思路**：
- 旧 key/prototype 形成锚定矩阵
- 新任务更新只允许在不改变旧任务最近邻关系或 top-k 排序的方向上移动
- prototype alignment loss 单独反传
- 对旧 prototype 建立稳定子空间 S_old

**导师建议（已采纳）**：
- ❌ 放弃「不改变最近邻关系」的不可计算约束
- ✅ **Key Relation Distillation Loss**：
  - 保存旧任务所有类的 key prototype 矩阵 K_t
  - 计算旧 key 之间的 pairwise 相似度矩阵 S_t = K_t K_t^T
  - 新任务训练时加 L_key_rel = ||S_t - Ŝ_t||_F²（只约束相对几何结构，允许整体旋转/平移）
- ✅ **Alternating update 策略**（避免梯度冲突）：
  - Step 1: CE loss → 更新 expert/prompt/router（key 冻结）
  - Step 2: Alignment loss → 更新 key（其他冻结）

---

### 2.5 v1 版本实现记录（2025-07-15）

> **版本**：v1 — 三组件异构梯度保护框架落地
> **状态**：已实现，默认关闭（通过超参数控制）

#### 2.5.1 新增文件结构

```
SMoPE/
├── protection/                    # ← 新增模块
│   ├── __init__.py                #   模块入口，导出所有保护函数
│   ├── task_memory.py             #   TaskMemory 跨任务持久化数据结构
│   ├── router_kl.py               #   组件一：Router KL 散度正则
│   ├── gradient_projection.py     #   组件二：SplitLoRA 式梯度投影
│   └── key_relation.py            #   组件三：Key Relation Distillation Loss
```

#### 2.5.2 修改的现有文件

| 文件 | 变更内容 |
|------|---------|
| `models/zoo.py` | `OnePrompt` 类新增 v1 API 接口：`get_router_and_input()`, `get_all_expert_keys()`, `get_key_query()`, `freeze_keys()`, `unfreeze_keys()`, `freeze_experts()`, `unfreeze_experts()`, `get_expert_param_groups()`, `get_v1_config()` |
| `learners/prompt.py` | `OnePrompt` learner 集成三组件保护机制：`update_model()` 中添加 KL 散度正则、梯度投影、Key Relation Loss + alternating update；新增 `_on_task_finish()` 方法在每任务完成后保存约束信息 |

#### 2.5.3 组件实现细节

**组件一：Router KL 散度正则** (`protection/router_kl.py`)
- `compute_router_kl_loss()`: 对旧任务每类的 router prototype 计算 KL(P_old || P_cur)
- `save_router_prototypes()`: 任务完成后 per-class 平均 router logits + input representation
- 备选 `compute_router_l2_loss()`: 当 KL 不稳定时的 L2 退化方案

**组件二：梯度投影** (`protection/gradient_projection.py`)
- `estimate_global_major_subspace()`: 对所有 expert 梯度拼接后做 SVD，取 95% 方差
- `project_gradients_to_minor_subspace()`: 梯度 = g - α·P_major(g)，α 由 expert 使用频率决定
- `collect_expert_gradients()`: 收集所有 e_pk/e_pv 参数的梯度
- 支持随机 SVD 近似（`_randomized_svd()`）用于大矩阵加速

**组件三：Key Relation Distillation** (`protection/key_relation.py`)
- `compute_key_relation_loss()`: L = ||S_t - Ŝ_t||_F²，约束 pairwise 相似度结构
- `compute_prototype_alignment_loss()`: L = ||K_t - K̂_t||_F²，约束绝对位置
- `save_key_prototypes()`: 保存 per-class key query 均值和归一化相似度矩阵

#### 2.5.4 超参数控制

所有 v1 保护机制默认关闭（权重为 0），通过模型 `get_v1_config()` 返回的超参数控制：

| 超参数 | 默认值 | 含义 |
|--------|-------|------|
| `lambda_router` | 0.0 | Router KL 散度正则权重 |
| `lambda_key` | 0.0 | Key Relation Distillation 权重 |
| `lambda_proto` | 0.0 | Prototype Alignment 权重 |
| `use_grad_projection` | False | 是否启用梯度投影 |
| `use_alternating_update` | False | 是否启用交替更新 |
| `freq_threshold` | 0.1 | Expert 使用频率阈值 |
| `temperature` | 1.0 | KL 散度温度参数 |

#### 2.5.5 已知限制 & 后续改进

| 限制 | 等级 | 计划 |
|------|------|------|
| 超参数需手动修改 `get_v1_config()` 返回值，未接入 YAML 配置 | 中 | v2 接入 config yaml + CLI args |
| SVD 在每任务结束时重算全部样本梯度，大 d_total 下开销大 | 中 | 改用随机 SVD 或 incremental SVD |
| Key Relation Loss 的 key prototype 使用 cls_token query 近似 | 低 | 验证与实际 e_pk 几何关系的一致性 |
| Alternating update 每 batch 创建新 optimizer，效率低 | 低 | 复用 optimizer 或每 N batch 执行一次 |
| 多 GPU (DataParallel) 下的 old_memories 访问需验证 | 低 | 添加 module 穿透的测试 |

#### 2.5.6 启用方式

```python
# 在 OnePrompt.get_v1_config() 中修改返回值，或在 learner 初始化后设置：
# learner._v1_config["lambda_router"] = 0.1
# learner._v1_config["use_grad_projection"] = True
# learner._v1_config["use_alternating_update"] = True
```

---

### 2.6 v2 版本优化记录（2025-07-15）

> **版本**：v2 — 诊断日志 + NaN 修复 + 增量 SVD + 温度软化
> **状态**：已实现，默认启用（超参数已优化）

#### 2.6.1 v1 训练数据分析

在 CIFAR-100 10-task 上训练 v1（组件全开，5 repeats），与原始 SMoPE baseline 对比：

| 指标 | Baseline | v1 | Δ |
|------|----------|----|---|
| FAA | 88.88 | 88.83 | **-0.05**（退化） |
| CAA | 92.78 | 92.81 | +0.03 |
| FR | 4.30 | 4.06 | -0.24（遗忘略减） |

逐任务准确率从 Task 3 开始 v1 持续落后，差距随任务数增大（Task 10: 88.88→88.64）。

**三个关键 Bug 定位**：

| # | 现象 | 根因 | 影响 |
|---|------|------|------|
| 1 | Loss NaN ×900 次 | `softmax→log(0)=-inf`，`lambda_key=0.5` 过大约束 | 组件一/三损失无效 |
| 2 | SVD r=1（d=230400） | 单任务末期梯度方向高度共线 | 组件二梯度投影 ≈ 恒等映射 |
| 3 | Router prototype 保存失败 | `(128×3) @ (64×25)` 维度不匹配 | replay-1 task-0 的 KL/proto 数据损坏 |

#### 2.6.2 新增文件

```
SMoPE/
├── protection/
│   └── loss_logger.py             # ← v2 新增：DiagnosticLogger 诊断日志模块
```

#### 2.6.3 修改的现有文件

| 文件 | 变更内容 |
|------|---------|
| `protection/router_kl.py` | P0: `softmax` 后 `clamp(min=1e-8)` + `renormalize` 防 NaN；新增 per-memory NaN 检测跳过；新增 `compute_router_kl_with_fallback()` 自动退化到 L2 |
| `protection/gradient_projection.py` | P1: 新增 `IncrementalSubspaceEstimator` 类（跨任务梯度快照缓冲区 + 中心化 SVD + `min_rank=5`）；`estimate_global_major_subspace()` 加中心化和 `min_rank` 参数 |
| `protection/key_relation.py` | P2: `compute_key_relation_loss()` 新增 `temperature` 参数（默认 2.0），软化 pairwise 相似度矩阵 |
| `protection/__init__.py` | 导出新增符号：`IncrementalSubspaceEstimator`, `compute_router_kl_with_fallback`, `DiagnosticLogger`, `compute_key_sim_distance` |
| `models/zoo.py` | P0: 超参数降权；新增 `key_temperature` 和 `enable_diagnostic_log` 配置项 |
| `learners/prompt.py` | P0+P1+P2: `update_model()` 分离各 loss 分量 + NaN 自动退化；`_on_task_finish()` 使用 IncrementalSubspaceEstimator + 记录 SVD 谱/expert 频率/key sim 距离；集成 DiagnosticLogger |
| `pull.md` | 追加 v2 PR 条目（Bug 分析 + P0/P1/P2 修改对照表） |

#### 2.6.4 组件优化细节

**P0 — 紧急修复（组件一 NaN + 超参数降权）**

- `compute_router_kl_loss()`: `cur_probs.log()` → `cur_probs.clamp(min=eps).log()`，重新归一化保证概率和为 1
- 新增 `compute_router_kl_with_fallback()`：先尝试 KL，NaN 时自动切换 L2，两层都失败返回 0
- 超参数降权：`lambda_router: 0.1→0.01`, `lambda_key: 0.5→0.05`, `lambda_proto: 0.05→0.01`
- 理由：v1 的 λ 过大导致约束压倒 CE loss，组件三实际是唯一能工作的（但约束过度）

**P1 — 增量 SVD（组件二改造）**

- 新增 `IncrementalSubspaceEstimator` 类
  - 维护跨任务梯度快照缓冲区（FIFO, buffer_size=200）
  - SVD 前对梯度矩阵做**中心化**（减去均值），使 SVD 捕获变化方向而非均值方向
  - `min_rank=5` 硬约束：SVD 保留的秩至少为 5，解决单任务 r=1 退化
  - 存储到 CPU 以节省显存
- `estimate_global_major_subspace()` 同步加中心化和 `min_rank` 参数
- 理由：v1 每任务独立 SVD，末期梯度几乎共线 → r=1；v2 跨任务累积 + 中心化 → r≥5

**P2 — 诊断日志 + 温度软化（组件三）**

- 新建 `DiagnosticLogger`（`protection/loss_logger.py`）：
  - `log_losses()`: 每 N batch 记录 L_ce / L_kl / L_key_rel / L_proto / L_total，标注 NaN 和退化方案
  - `log_svd_spectrum()`: 记录 top-10 奇异值 + 累计方差比例 + condition number + effective rank
  - `log_expert_freqs()`: expert 频率直方图 + top/bottom-5 + entropy
  - `log_key_sim_distance()`: 旧任务 pairwise 相似度距离（Frobenius/Max/Mean）
  - 日志路径：`outputs/cifar-100/10-task/one-prompt/lossoutput.log`
- `compute_key_relation_loss()` 加 `temperature=2.0`：`softmax(logits/T)` 软化分布，降低 pairwise sim 约束的尖锐度
- `save_router_prototypes()` 保存 `router_pairwise_sim` 时同步使用 `key_temperature`

#### 2.6.5 超参数变更对照

| 超参数 | v1 默认值 | v2 默认值 | 说明 |
|--------|----------|----------|------|
| `lambda_router` | 0.1 | **0.01** | KL 散度权重降 10× |
| `lambda_key` | 0.5 | **0.05** | Key Relation 权重降 10× |
| `lambda_proto` | 0.05 | **0.01** | Prototype Alignment 权重降 5× |
| `key_temperature` | — | **2.0** | 新增：Key Relation 温度 |
| `enable_diagnostic_log` | — | **True** | 新增：启用分项 loss 日志 |
| `freq_threshold` | 0.1 | 0.1 | 不变 |
| `temperature` | 1.0 | 1.0 | KL 温度不变 |
| `use_grad_projection` | True | True | 不变 |
| `use_alternating_update` | True | True | 不变 |

#### 2.6.6 v2 训练期数据流

```
每个 batch:
  L_ce → L_kl(NaN?→L2) → L_proto → total_loss.backward()
  → 梯度投影(min_rank≥5) → optimizer_ce.step()
  → L_key_rel(temperature=2.0) → optimizer_key.step()
  → DiagnosticLogger.log_losses()

每个 task 完成:
  save_router_prototypes(temperature=2.0)
  → IncrementalSubspaceEstimator.add_snapshots(跨任务梯度)
  → estimate_subspace(min_rank=5) → 记录 SVD 谱
  → 记录 expert 频率分布 + key sim 距离
```

#### 2.6.7 已知限制 & 后续改进

| 限制 | 等级 | 计划 |
|------|------|------|
| IncrementalSubspaceEstimator 缓冲区使用 CPU 存储，大 d_total 下内存占用可观 | 中 | v3: 改用随机投影压缩梯度快照（[n, d]→[n, k] where k≪d） |
| 超参数仍为手动设定，未做 systematic hyperparameter search | 中 | v3: 每个 λ 做 grid search 或 Bayesian optimization |
| DiagnosticLogger 间隔固定 10 batch，大任务下日志量大 | 低 | 改为自适应间隔（early epoch 密集、later epoch 稀疏） |
| Alternating update 每 batch 创建新 key_optimizer | 低 | 复用 optimizer 实例，zero_grad 替代重建 |

---

### 2.7 v3 版本优化记录（2025-07-15）

> **版本**：v3 — 直接权重空间正则 + 特征蒸馏
> **状态**：已实现，默认启用（三组件全部重写）
> **分支**：`v3`

#### 2.7.1 v2 训练数据分析

在 CIFAR-100 10-task 上训练 v2（5 repeats × 2 实验），与原始 SMoPE baseline 对比：

| 实验 | FAA | CAA | FR | Δ FAA |
|------|-----|-----|-----|-------|
| Baseline (guide.md) | 88.88 | 92.78 | 4.30 | — |
| v2 Exp1 (5 trials) | 88.83 | 92.78 | 4.06 | **-0.05** |
| v2 Exp2 (5 trials) | 88.83 | 92.81 | 4.06 | **-0.05** |

**诊断日志关键发现**（基于 `lossoutput.log` 845 个采样点的分析）：

| # | 现象 | 根因 | 影响 |
|---|------|------|------|
| 1 | **所有保护损失恒为零** | 组件一 KL ≈ 0（router 自然稳定），退化 L2 也 ≈ 0 | 三个组件对训练无任何正则化作用 |
| 2 | **梯度维度塌缩** 230400→110 | T2 起 `collect_expert_gradients` 只收集到 110 维 | 组件二 SVD 在错误空间投影，可能损坏梯度 |
| 3 | 702/845 batch KL 退化到 L2 | `compute_router_kl_with_fallback` 全覆盖 | 组件一退化为几乎零损失的 L2 |

**根因分析**：三个组件都在保护 **router 的输出行为**（logits、pairwise 相似度、梯度子空间），但 router（e_pk 参数 [1, 64] per expert）极其稳定——一旦训练完成几乎不漂移。保护不漂移的对象自然产生零损失。而真正导致遗忘的 **e_pv（expert value）参数漂移**完全未被约束。

#### 2.7.2 修改的现有文件

| 文件 | 变更内容 |
|------|---------|
| `protection/router_kl.py` | **重写**：删除 `compute_router_kl_loss` / `compute_router_kl_with_fallback`（KL + fallback）；新增 `save_pk_weights()` 保存 e_pk 参数快照、`compute_pk_l2_reg()` 计算 usage-frequency-weighted L2 正则、`save_pv_proto_outputs()` 保存 e_pv 输出特征（→ 组件三）、`_compute_pv_features()` 内部辅助函数 |
| `protection/gradient_projection.py` | **重写**：删除 SVD 全链路（`IncrementalSubspaceEstimator` 变空壳、`estimate_global_major_subspace` / `project_gradients_to_minor_subspace` 标记废弃）；新增 `save_pv_weights()` 保存 e_pv 参数快照、`compute_pv_l2_reg()` 计算 usage-frequency-weighted L2 正则（**主保护**）；`collect_expert_gradients` 保留但加维度断言 |
| `protection/key_relation.py` | **重写**：删除 `compute_key_relation_loss` / `compute_prototype_alignment_loss`（pairwise 相似度）；新增 `compute_feature_distill_loss()` 对旧类 prototype 的 e_pv 输出特征做 MSE 蒸馏 |
| `protection/loss_logger.py` | **更新**：`log_losses()` 标签改为 `L_pk`/`L_pv`/`L_feat`；新增 `log_task_finish()` 记录权重漂移 + 特征漂移；新增 `compute_weight_drift()` / `compute_feature_drift_for_memory()` 便捷函数；`log_svd_spectrum` / `log_key_sim_distance` 标记废弃 |
| `protection/task_memory.py` | **更新**：新增 `pk_snapshot`（dict）、`pv_snapshot`（dict）、`pv_proto_outputs`（tensor）字段；`global_major_subspace` / `grad_matrix` 保留但不再填充 |
| `protection/__init__.py` | **更新**：导出新符号 `save_pk_weights`, `save_pv_proto_outputs`, `compute_pk_l2_reg`, `save_pv_weights`, `compute_pv_l2_reg`, `compute_feature_distill_loss`；移除旧导出 |
| `learners/prompt.py` | **重写**：`update_model()` 删除交替更新（freeze/unfreeze 逻辑）、删除 Router KL / Prototype Alignment / Key Relation 相关代码；改为单一 backward pass 计算 `L_total = L_ce + λ_pk·L_pk + λ_pv·L_pv + λ_feat·L_feat`；`_on_task_finish()` 删除 SVD 梯度估计、删除 key sim 距离计算；改为保存 e_pk/e_pv 权重快照、e_pv proto outputs、诊断日志（权重漂移 + 特征漂移） |
| `models/zoo.py` | **更新**：`get_v1_config()` 返回值改为 `lambda_pk=0.01`, `lambda_pv=0.05`, `lambda_feat=0.01`；移除 `lambda_router` / `lambda_key` / `lambda_proto` / `use_grad_projection` / `use_alternating_update` |

#### 2.7.3 组件实现细节

**P0 — 组件一重写：e_pk 权重空间 L2 正则** (`protection/router_kl.py`)

- `save_pk_weights(prompt)`: 遍历所有 `e_pk` 参数，保存 `.detach().cpu().clone()` 到 dict
- `compute_pk_l2_reg(prompt, old_memories)`: 
  - 对每个旧任务的 pk_snapshot，计算 `weight · MSE(p_cur, p_old)`
  - 权重由 `_get_pk_expert_weight()` 解析参数名中的 expert 索引，查 usage_freq 得到
  - `freq_threshold=0` 表示全部 expert 都约束（按频率加权）
  - 返回值除以参数数量做归一化
- `_compute_pv_features(prompt, x_query)`: 计算所有 expert 的 query @ e_pv 点积（供组件三使用）

**P1 — 组件二重写：e_pv 权重空间 L2 正则** (`protection/gradient_projection.py`)

- `save_pv_weights(prompt)`: 遍历所有 `e_pv` 参数保存快照
- `compute_pv_l2_reg(prompt, old_memories)`: 与组件一结构相同，作用于 e_pv
- 这是 v3 的**核心保护机制**：直接约束高频 expert 的 value 参数不漂移
- `save_expert_usage_freqs()` 保留不变

**P2 — 组件三重写：e_pv 特征蒸馏** (`protection/key_relation.py`)

- `compute_feature_distill_loss(prompt, old_memories)`:
  - 对每个旧任务，取出保存的 `pv_proto_outputs`（各类 prototype 的 e_pv 输出特征）
  - 用当前 e_pv 参数重新计算同类 prototype 的特征 → MSE
  - 与组件二互补：组件二约束参数空间，组件三约束功能空间
- `save_pv_proto_outputs(model, dataloader)`: 通过 ViT patch_embed + `_compute_pv_features` 计算 per-class 均值特征
  - 处理 DataParallel 包装：`_model = model.module if hasattr(model, 'module') else model`

#### 2.7.4 超参数变更对照

| 超参数 | v2 默认值 | v3 默认值 | 说明 |
|--------|----------|----------|------|
| `lambda_router` | 0.01 | — **移除** | 替换为 `lambda_pk` |
| `lambda_pk` | — | **0.01** | 新增：e_pk 权重空间 L2 正则权重 |
| `lambda_key` | 0.05 | — **移除** | 替换为 `lambda_pv` |
| `lambda_pv` | — | **0.05** | 新增：e_pv 权重空间 L2 正则权重（主保护） |
| `lambda_proto` | 0.01 | — **移除** | 替换为 `lambda_feat` |
| `lambda_feat` | — | **0.01** | 新增：e_pv 特征蒸馏权重 |
| `freq_threshold` | 0.1 | **0.0** | 改为 0（全部 expert 按频率加权约束） |
| `use_grad_projection` | True | — **移除** | 梯度投影已废弃 |
| `use_alternating_update` | True | — **移除** | 交替更新已废弃（单一 backward pass） |
| `key_temperature` | 2.0 | 2.0 | 保留兼容 |
| `enable_diagnostic_log` | True | True | 不变 |

#### 2.7.5 v3 训练期数据流

```
每个 batch:
  L_ce → L_pk (freq-weighted e_pk L2) → L_pv (freq-weighted e_pv L2)
  → L_feat (e_pv feature distill on old prototypes)
  → total_loss.backward() → optimizer.step()
  → DiagnosticLogger.log_losses(L_ce, L_pk, L_pv, L_feat, L_total)

每个 task 完成:
  save_pk_weights() → memory.pk_snapshot
  save_pv_weights() → memory.pv_snapshot
  save_router_prototypes() → memory.input_prototypes
  save_pv_proto_outputs() → memory.pv_proto_outputs
  save_expert_usage_freqs() → memory.expert_usage_freq
  → DiagnosticLogger.log_expert_freqs()
  → compute_weight_drift(pk, pv) → DiagnosticLogger.log_task_finish(pk_drift, pv_drift, feat_drifts)
```

#### 2.7.6 已知限制 & 后续改进

| 限制 | 等级 | 计划 |
|------|------|------|
| `_compute_pv_features` 使用全部 expert 而非 top-K，与 SMoPE 实际 prompt attention 计算不完全一致 | 低 | 组件二（权重 L2）是主保护，组件三是辅助；如需精确，可改为保存 CLS token |
| λ 超参数为手动设定，未做 grid search | 中 | v4: Bayesian optimization 或每个 λ 做 sweep |
| `compute_pk_l2_reg` / `compute_pv_l2_reg` 每 batch 遍历所有旧任务的每个参数，旧任务多时开销增大 | 中 | 合并旧任务快照（running average）或每 N batch 计算一次 |
| 权重漂移诊断在任务完成时遍历所有参数，大模型下耗时 | 低 | 采样部分参数或异步计算 |
| 未接入 YAML 配置文件 | 低 | 后续接入 config yaml + CLI args |

#### 2.7.7 启用方式

```python
# v3 默认启用（lambda_pk=0.01, lambda_pv=0.05, lambda_feat=0.01）
# 修改超参数：编辑 models/zoo.py 中 OnePrompt.get_v1_config() 的返回值
# 或在 learner 初始化后动态设置：
# learner._v1_config["lambda_pv"] = 0.1
# learner._v1_config["lambda_feat"] = 0.0  # 关闭组件三
```

---

### 2.8 v3-lite 版本优化记录（2026-06-25）

> **版本**：v3-lite — 保留三组件思路，改为轻量锚点正则 + 近邻特征蒸馏 + 独立有效数据日志
> **状态**：已实现，默认启用；目标是在不显著拖慢速度的前提下争取约 1%-2% 的性能提升。
> **选择依据**：v1 存在 NaN 与 SVD r=1 退化；v2 在 FAA 上基本无增益；v3 的方向更接近遗忘根因（e_pv 漂移），但逐旧任务逐参数计算和全量特征蒸馏导致速度过慢。

#### 2.8.1 数据分析结论

基于 `output.log` 第 8652 行以后、`v3_output.log`、`v3_lossoutput.log` 的对比：

| 版本 | 主要现象 | 结论 |
|------|---------|------|
| v1 | 训练中出现 NaN，SVD 子空间经常退化到 r=1 | 不适合作为继续优化基线 |
| v2 | FAA 与 baseline 基本持平，保护 loss 大量接近 0 | 稳定性修复有效，但保护信号太弱 |
| v3 | e_pk/e_pv/feature loss 有非零信号，但验证与训练耗时明显升高 | 保留方向，压缩实现成本 |

关键判断：router/key 侧本身较稳定，继续强约束 e_pk 收益有限；遗忘更可能来自 e_pv value 侧漂移。因此 v3-lite 降低 e_pk 约束，增强 e_pv 约束，并把 feature distill 作为低成本辅助项。

#### 2.8.2 修改的现有文件

| 文件 | 变更内容 |
|------|---------|
| `protection/router_kl.py` | 新增 `build_pk_l2_anchor()`、`compute_pk_l2_reg_from_anchor()`；新增快速版 `_compute_pv_features()`，按 head 批量计算所有 expert，替代大量小矩阵乘法 |
| `protection/gradient_projection.py` | 新增 `build_pv_l2_anchor()`、`compute_pv_l2_reg_from_anchor()`，将 e_pv 逐旧任务 L2 改为加权锚点 L2 |
| `protection/key_relation.py` | `compute_feature_distill_loss()` 新增 `max_memories`，默认只对最近若干旧任务做功能空间蒸馏 |
| `protection/__init__.py` | 导出新增 anchor 构建与 anchor 正则函数 |
| `learners/prompt.py` | 新增 `_refresh_l2_anchors()`；训练时使用 anchor 正则；task 结束后刷新 anchor；组件三直接用 `input_protos` 生成 `pv_proto_outputs`，避免重复遍历 dataloader 和 prototype 不一致 |
| `models/zoo.py` | 新增 v3-lite 默认超参数：`lambda_pk=0.003`, `lambda_pv=0.12`, `lambda_feat=0.02`, `freq_threshold=0.02`, `max_feature_memories=4`，默认关闭高频诊断日志 |
| `run.py` | 主输出改为 `v3_lite_output.log`；新增 `EffectiveDataLogger`，输出独立有效数据日志 `v3_lite_effective.log` |
| `experiments/cifar-100_v3_lite.sh` | 新增 CIFAR-100 v3-lite 启动脚本 |

#### 2.8.3 三组件实现逻辑

**组件一：e_pk 弱锚点正则**

原 v3 每个 batch 对所有旧任务的 `pk_snapshot` 逐参数计算：

```text
sum_t w_t * MSE(p_cur, p_old_t)
```

v3-lite 利用等价梯度形式，把多个旧任务快照合并为一个加权 anchor：

```text
anchor = sum_t w_t * p_old_t / sum_t w_t
loss   = sum_t w_t * MSE(p_cur, anchor)
```

该形式与原式对当前参数 `p_cur` 的梯度一致，只去掉常数项，训练时不再随旧任务数量线性增长。由于日志显示 router/key 较稳定，默认 `lambda_pk` 降为 `0.003`。

**组件二：e_pv 主保护锚点正则**

e_pv 是 v3-lite 的主保护对象。实现与 e_pk 相同，但默认权重提高到 `lambda_pv=0.12`。同时 `freq_threshold=0.02`，低频 expert 不进入锚点约束，保留新任务塑性。

**组件三：近邻功能空间蒸馏**

v3 原实现对所有旧任务 prototype 反复计算 `_compute_pv_features()`，且保存 `pv_proto_outputs` 时使用的 query 与训练时的 `input_prototypes` 存在不一致。v3-lite 做两点修正：

- task 结束时直接用同一组 `input_protos` 生成 `pv_proto_outputs`
- 训练时只对最近 `max_feature_memories=4` 个旧任务做 feature distill

这样组件三仍保护 e_pv 在旧类 prototype 上的函数行为，但成本不会随 10-task 全量历史快速膨胀。

#### 2.8.4 超参数变更对照

| 超参数 | v3 默认值 | v3-lite 默认值 | 说明 |
|--------|----------|---------------|------|
| `lambda_pk` | 0.01 | **0.003** | e_pk 已较稳定，弱约束即可 |
| `lambda_pv` | 0.05 | **0.12** | e_pv 漂移是主保护目标 |
| `lambda_feat` | 0.01 | **0.02** | 增强功能空间辅助约束 |
| `freq_threshold` | 0.0 | **0.02** | 低频 expert 不约束，保留塑性 |
| `max_feature_memories` | 无 | **4** | 特征蒸馏只使用最近旧任务 |
| `enable_diagnostic_log` | True | **False** | 默认关闭高频 batch 诊断，避免 I/O 拖慢 |
| `diagnostic_log_interval` | 10 | **50** | 若开启诊断，默认更稀疏 |

#### 2.8.5 v3-lite 训练期数据流

```text
每个 task 结束:
  save_pk_weights / save_pv_weights
  save_router_prototypes -> input_prototypes
  _compute_pv_features(input_prototypes) -> pv_proto_outputs
  save_expert_usage_freqs
  build_pk_l2_anchor + build_pv_l2_anchor

每个 batch:
  L_ce
  + lambda_pk * L_pk(anchor)
  + lambda_pv * L_pv(anchor)
  + lambda_feat * L_feat(recent old prototypes)
  -> single backward -> optimizer.step()
```

#### 2.8.6 输出与有效数据日志

主训练输出文件改为：

```text
outputs/cifar-100/10-task/one-prompt/v3_lite_output.log
```

整体精度数据项保持不变，仍输出并保存：

```text
acc, time, fr, FAA, CAA, FR
```

另新增单独有效数据日志：

```text
outputs/cifar-100/10-task/one-prompt/v3_lite_effective.log
```

该日志为 JSON lines，每个 repeat 记录：

- `faa_by_task`
- `fr_by_task`
- `time_per_epoch_by_task`
- `final_per_task_acc`
- `forgetting_by_task`
- `worst_final_task_id`
- `max_forgetting_task_id`
- `late_task_plasticity`
- `running_summary`

用途：下次优化时无需重新解析完整 stdout，可直接定位是“旧任务遗忘”“新任务塑性不足”还是“速度瓶颈”。

#### 2.8.7 启动方式

```bash
bash experiments/cifar-100_v3_lite.sh
```

验证记录：

```text
python -m py_compile run.py learners/prompt.py protection/router_kl.py protection/gradient_projection.py protection/key_relation.py protection/__init__.py models/zoo.py
python test_device_fix.py  # 在 UTF-8 输出环境下通过；普通 GBK 控制台可能因 ✓ 字符打印失败
```

---

### 2.9 v4-light / v4-split-lite 版本优化记录（2026-06-26）

> **版本**：v4-split-lite — 轻量 router 稳态 + expert anchor + SplitLoRA-style 在线低秩软投影  
> **状态**：已实现并推送到 `v4` 分支。  
> **核心判断**：v4-light 可以作为工程止血版本，但如果项目创新点仍要落在“把 SplitLoRA 思想迁移到 SMoPE”，则必须保留梯度空间切分/投影这一机制。v4-split-lite 因此补回轻量化的 major-subspace soft projection，同时避免 v1/v2 的重型全量 SVD。

#### 2.9.1 方向修正结论

v1/v2/v3/v3-lite 的实验说明：

| 版本 | 主要问题 | 处理策略 |
|------|---------|---------|
| v1 | NaN、SVD 退化到 r=1、保护信号不稳定 | 不再沿用重型 per-task SVD 实现 |
| v2 | 修复稳定性后 FAA 仍基本无增益，KL/key loss 大量接近 0 | 不把 router/key KL 作为主创新机制 |
| v3 | e_pk/e_pv/feature loss 有信号，但速度显著下降 | 保留 e_pv 是遗忘主因的判断，压缩成本 |
| v3-lite | 完整 10 task FAA 约 88.99，提升有限且仍慢 | 作为对照，不作为最终 SplitLoRA 叙事版本 |
| v4-light | 速度友好，但与 SplitLoRA 的梯度空间切分关系偏弱 | 升级为 v4-split-lite |

最终选择：保留三组件设计，但把组件二重新拉回 SplitLoRA 语义，即“旧任务 major direction 软削弱，新任务保留 minor-space 可塑性”。

#### 2.9.2 三组件实现逻辑

**组件一：Router / Gating 轻量稳态**

- 复用 SMoPE 已有 `prompt_scores`，不额外前向。
- 新增 router usage balance，避免专家选择坍缩到少数 expert。
- 新增 old usage prior，任务切换后轻微约束当前 routing 不偏离旧任务高频使用结构。
- 对应文件：`models/zoo.py`。

**组件二：Expert Value 的 SplitLoRA-style 在线低秩软投影**

- 新增 `protection/split_lite.py`。
- 默认只作用于 `e_pv`，因为日志和 v3 诊断表明 e_pv 漂移更可能是遗忘主因。
- 每隔 `split_lite_interval=20` 个 batch 采样一次当前 expert 梯度。
- 每个任务结束时，对高频 expert 的梯度样本做小 rank SVD，维护旧任务 major basis。
- 新任务训练时执行软投影：

```text
g_new <- g_new - alpha * P_major(g_new)
```

其中默认 `rank=4`，`alpha=0.2`，只削弱 major 方向而不是硬删除，避免新任务塑性被破坏。

**组件三：Key / Prototype 保留为轻量诊断与可选约束**

- 默认 `lambda_feat=0.0`，不再每 batch 做旧 prototype feature distillation。
- 保留 TaskMemory、anchor 与 feature distill 接口，方便后续消融或针对性开启。
- 当前主创新不再依赖昂贵的 feature distill，而是依赖组件二的低秩梯度空间保护。

#### 2.9.3 修改的文件

| 文件 | 变更内容 |
|------|---------|
| `protection/split_lite.py` | 新增 `SplitLiteProjector`，实现 per-expert gradient buffer、低秩 basis 构建、soft projection、投影诊断日志 |
| `protection/__init__.py` | 导出 `SplitLiteProjector` |
| `models/zoo.py` | 默认配置升级为 v4-split-lite：`use_split_lite=True`, `split_lite_rank=4`, `split_lite_alpha=0.2`, `split_lite_interval=20`, only `e_pv` |
| `learners/prompt.py` | 在 `total_loss.backward()` 后、`optimizer.step()` 前调用 split-lite 投影；任务结束时按 expert usage 更新 basis |
| `run.py` | 主输出改为 `v4_split_lite_output.log`；有效数据日志改为 `v4_split_lite_effective.log`；新增 `--max_task` 命令行覆盖项 |
| `experiments/cifar-100_v4_split_lite.sh` | 新增 CIFAR-100 v4-split-lite 启动脚本 |

#### 2.9.4 默认超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `lambda_pk` | 0.0005 | e_pk 弱 anchor，避免过度约束 router key |
| `lambda_pv` | 0.015 | e_pv 弱 anchor，作为 soft projection 的辅助保护 |
| `lambda_feat` | 0.0 | 默认关闭昂贵 feature distill |
| `route_balance_weight` | 2e-4 | 防止 expert usage 坍缩 |
| `route_prior_weight` | 1e-4 | 轻微贴近旧任务 expert usage prior |
| `use_split_lite` | True | 启用在线低秩软投影 |
| `split_lite_components` | `("e_pv",)` | 默认只保护 e_pv |
| `split_lite_rank` | 4 | 每个 expert 的 major basis rank |
| `split_lite_alpha` | 0.2 | major 方向削弱强度 |
| `split_lite_interval` | 20 | 每 20 batch 投影/采样一次 |
| `split_lite_buffer_size` | 24 | 每个 expert 保存的梯度样本数 |
| `split_lite_expert_threshold` | 0.03 | 只为高频 expert 建 basis |
| `split_lite_basis_decay` | 0.7 | 合并旧 basis 与新梯度样本时旧 basis 的保留强度 |

#### 2.9.5 输出与有效数据日志

主精度输出：

```text
outputs/cifar-100/10-task/one-prompt/v4_split_lite_output.log
```

有效数据日志：

```text
outputs/cifar-100/10-task/one-prompt/v4_split_lite_effective.log
```

投影诊断日志：

```text
outputs/cifar-100/10-task/one-prompt/v4_split_lite_projection.log
```

其中 `v4_split_lite_effective.log` 继续保留整体精度数据项，并额外记录：

- `version = v4_split_lite`
- `early_mean_faa`
- `late_mean_faa`
- `late_task_plasticity`
- `aux_logs.projection`

`v4_split_lite_projection.log` 记录每个任务结束后的 basis 规模、active experts、projected vectors、collected vectors，用于判断投影是否真的发生。

#### 2.9.6 运行时间预估

用户指定配置：

```text
REPEAT=1
MAX_TASK=10
CRCT_EPOCHS=50
rank=4
alpha=0.2
interval=20
components=e_pv
```

参考 `v3_lite_effective.log`：CIFAR-100 10-task、repeat=1、CRCT_EPOCHS=50 的总时长约 `5:51:16`。v4-split-lite 默认只对 `e_pv` 做 rank-4、每 20 batch 一次的软投影，预计额外开销约 3%-8%。

因此预估总时长：

```text
约 6.0 到 6.4 小时
保守上界：约 7 小时
```

如果 `torch` 首次加载、磁盘 IO、GPU 占用或 CUDA/cuDNN 状态不同，实际时长会有波动。

#### 2.9.7 调参指南：先少跑，再放大

**阶段 A：快速筛参**

```bash
REPEAT=1
MAX_TASK=5
CRCT_EPOCHS=10 或 20
```

目的：确认代码稳定、projection log 中有 basis 和 projected vectors、前 5 task FAA 不明显下降。

优先观察：

- `v4_split_lite_projection.log` 中 `basis_sizes` 是否非空
- `projected_vectors` 是否大于 0
- `early_mean_faa` 是否接近 v4-light / v3-lite

**阶段 B：半量确认**

```bash
REPEAT=3
MAX_TASK=10
CRCT_EPOCHS=20 或 30
```

目的：判断趋势是否稳定，重点看 `late_mean_faa`、最终 FAA、FR。

**阶段 C：正式实验**

```bash
REPEAT=5
MAX_TASK=-1
CRCT_EPOCHS=50
```

目的：形成可写入论文/报告的完整对比结果。

**推荐 sweep 顺序**

1. 固定 `rank=4`, `interval=20`，扫 `alpha = 0.1, 0.2, 0.3`
2. 若遗忘仍高，尝试 `rank=8`
3. 若速度慢，先把 `interval=20` 改为 `40`
4. 若新任务塑性下降，降低 `alpha` 或提高 `split_lite_expert_threshold`
5. 暂不建议默认开启 `e_pk` 投影；只有当 router drift 诊断显示明显问题时再加

#### 2.9.8 验证记录

```text
python -m py_compile run.py learners/prompt.py models/zoo.py protection/split_lite.py protection/__init__.py
```

额外 dummy prompt 测试通过：`SplitLiteProjector` 可构建 basis，并在下一任务执行 projected vectors。

---

### 2.10 v5-transient-prompt 版本设计记录（2026-06-30）

> **版本**：v5-transient-prompt — 在 v4-split-lite 上加入 CP-MoE 式瞬态 prompt 探针。  
> **核心判断**：不需要等 `split_lite_alpha / min_task / active_topk` 完整调完再加入瞬态 prompt。瞬态 prompt 改变的是任务开始前的 expert 兼容性估计和保护强度分配，会改变 v4 超参的最优区间；因此应先接入 v5，再做分阶段消融。

#### 2.10.1 机制定位

v5 不把 transient prompt 作为长期新增容量，而是作为每个新任务开始前的短 warm-up 探针：

1. 冻结 backbone、classifier 和稳定 SMoPE prompt，只临时更新 `e_pv`。
2. 用少量 batch 估计当前任务对各 expert 的 task-local 梯度重要性。
3. 立刻恢复原 `e_pv` 权重，丢弃瞬态更新。
4. 只保留 expert-level `cp_scores`，用于正式训练阶段。

这样保留 CP-MoE 的 assess-then-update 思想，同时不破坏 v4 的三组件主线。

#### 2.10.2 与 v4 三组件的融合方式

- **Router / Gating**：把 `cp_scores` 转成 centered bias，加到训练期 prompt expert top-k score 上，使新任务更倾向选择瞬态探针认为兼容的 expert。
- **Expert Value / Split-lite**：把 `cp_scores` 转成 expert-wise protection scale，同时调节 e_pv anchor 正则和 split-lite 投影强度。
- **Key / Prototype**：仍默认不启用昂贵 feature distill，只把 transient 结果写入独立日志和 TaskMemory，供后续诊断。

正式训练仍是 v4 的单阶段联合优化：`CE + router balance/prior + e_pv anchor`，并在 `backward()` 后、`optimizer.step()` 前执行 split-lite projection。瞬态 prompt 只在正式训练前单独 warm-up。

#### 2.10.3 新增文件与改动

| 文件 | 变更内容 |
|------|---------|
| `protection/transient_prompt.py` | 新增 `TransientPromptProbe`，实现短 warm-up、权重恢复、`cp_scores` 计算和 JSONL 日志 |
| `models/zoo.py` | `OnePrompt` 新增 transient CP score 缓存、router score bias、protection scale 接口 |
| `models/vit.py` | Attention prompt 输入支持可选 score bias |
| `protection/gradient_projection.py` | e_pv anchor 正则支持 expert-wise scale |
| `protection/split_lite.py` | split-lite projection 支持 expert-wise alpha scale |
| `learners/prompt.py` | 任务正式训练前运行 transient probe，并写入 TaskMemory |
| `run.py` / `trainer.py` | 新增 v5 CLI 参数；修复 `split_lite_active_topk` 未传入 learner config 的问题 |

#### 2.10.4 日志

v5 使用 `--experiment_version` 区分日志：

```text
{version}_output.log       # 主训练输出
{version}_effective.log    # FAA/CAA/FR/耗时摘要
{version}_projection.log   # split-lite 投影诊断
{version}_transient.log    # transient prompt 探针诊断
```

`{version}_transient.log` 每个任务记录 `warmup_batches`、`steps`、`mean_loss`、`cp_scores`、`importance_sum`。

#### 2.10.5 消融脚本

```text
experiments/cifar-100_v5_exp1_v4_delay_topk.sh      # v4 delayed/topk 对照
experiments/cifar-100_v5_exp2_transient_bias.sh     # 只加 transient router bias
experiments/cifar-100_v5_exp3_transient_full.sh     # router bias + e_pv protection scaling
```

推荐先跑 `MAX_TASK=5, REPEAT=5`，若 `v5_exp3` 的 CAA/FAA 趋势明显优于 `v5_exp1`，再扩展到 `MAX_TASK=10` 和正式 `CRCT_EPOCHS=50`。

2.11

# Sensitivity Basis 验证优先级

## Summary

不需要一上来从 3 个方向全做。建议按证据强弱和实现成本分层：

1. **先做旧类 prototype loss sensitivity basis**，这是最推荐的第一步。
2. **再做旧任务 validation loss sensitivity basis**，作为更真实但更贵的确认。
3. **function output drift 最大方向先不作为第一轮 basis**，更适合做辅助诊断。

核心原因：你现在要验证的是“split-lite 当前 basis 是否对准旧任务会遗忘的方向”。最直接的办法不是多跑大实验，而是比较两个子空间是否重合：

- 当前 split-lite basis：训练梯度 SVD 得到的方向。
- sensitivity basis：旧任务 loss/function 对 `e_pv` 最敏感的方向。

如果二者 overlap 很低，就说明 split-lite 投影的方向和真正需要保护的方向不是一回事。

## 三种方向怎么理解

**旧类 prototype loss 对 e_pv 的梯度：第一优先级**

这是最适合先做的版本。

机制是：每个旧类用一个 prototype 代表，例如旧类平均特征。然后计算当前模型在这些旧 prototype 上的 function/output drift 或 feature distill loss，再对 `e_pv` 反传，得到“旧类功能最怕变的方向”。

优点：

- 比 validation loss 快很多。
- 和 repo 里已有的 `input_prototypes / pv_proto_outputs / compute_feature_distill_loss` 思路一致。
- 很适合判断 split-lite basis 是否方向错。

缺点：

- prototype 是压缩代表，不等于完整旧任务数据。
- 如果 prototype 保存得不准，sensitivity 也会偏。

当前代码注意点：

- 现在默认 `lambda_feat=0.0`，所以 prototype 数据不一定保存。
- 需要加一个“只保存 prototype 用于诊断，但不启用 feature loss”的开关。

**旧任务 validation loss 对 e_pv 的梯度：第二优先级**

这是最真实的版本。

机制是：拿旧任务 validation loader，计算旧任务 CE loss，对 `e_pv` 反传，得到旧任务 accuracy/loss 真正敏感的梯度方向。

优点：

- 和 “旧任务 accuracy 会掉” 最直接相关。
- 证据最硬。

缺点：

- 成本高，需要保留或重新加载旧任务 validation 数据。
- 每个旧任务、每个 expert 做梯度收集会慢。
- 如果 batch 很少，方向会有噪声；如果 batch 多，运行时间会上去。

所以它适合作为 prototype 版本之后的确认实验，而不是第一步。

**function output drift 最大方向：辅助诊断，不建议第一轮作为主 basis**

这个方向容易误解。
“output drift 最大”本身是一个现象，不天然等于一个可投影 basis。你要把它变成 basis，仍然需要定义一个 loss，例如：

```
L_drift = || current_pv_output(old_proto) - saved_pv_output(old_proto) ||^2
```

然后对 `e_pv` 反传，这其实就回到了 prototype sensitivity basis。

所以它更适合做辅助判断：

- 如果某些旧类 prototype 的 function drift 很大，说明旧功能确实在漂。
- 再看这些 drift 对应的 gradient basis 是否和 split-lite basis overlap 低。

## 推荐实验顺序

第一阶段只做 prototype sensitivity overlap，不跑完整训练：

- 在任务结束时，对已有 split-lite basis `B_split` 做记录。
- 对旧任务 prototype loss 反传，收集每个 expert 的 `e_pv` gradient。
- 对这些 gradient 做 SVD 得到 `B_sens_proto`。
- 计算每个 expert 的 overlap：

```
overlap = || B_split @ B_sens_proto.T ||_F^2 / rank
```

判断：

- overlap 高，比如 >0.5：split-lite 方向大体对，问题可能是 alpha/rank/强度。
- overlap 低，比如 <0.2：split-lite basis 没对准旧任务敏感方向，继续调 usage/topk/alpha 意义有限。
- overlap 中间：需要看具体哪些 expert、哪些 task 低。

第二阶段再做 alpha/rank 小矩阵：

- `alpha=0 / 0.2 / 0.5`
- `rank=4 / 8`
- 固定 `active_topk=16`
- 固定 `old_union + nonstrict`

目标是验证：

- 如果 overlap 高但性能不涨，可能是强度不够。
- 如果 overlap 低且性能不涨，说明方向错。

第三阶段才做 validation loss sensitivity basis：

- 只挑 prototype overlap 最低、遗忘最严重的几个 task/expert。
- 不需要全量旧任务全量 expert。
- 用 validation loss 做确认，避免成本爆炸。

## 是否能有效验证

可以，但要注意它验证的是“机制假设”，不是直接证明最终 FAA 会涨。

它能回答：

- 当前 split-lite 投影方向和旧任务敏感方向是否一致？
- 如果不一致，为什么 usage mode 更集中也没用？
- 如果一致但性能不涨，是不是 alpha/rank 太弱？

它不能单独回答：

- 最终能不能稳定 +1 FAA。
- 新 sensitivity basis 加进训练后一定有效。

所以它是一个很好的“方向诊断实验”，不是最终性能实验。

## 建议结论

不要三个方向一起做。第一轮只做：

**旧类 prototype loss sensitivity basis vs current split-lite basis overlap**

这是最便宜、最贴近现有代码、最能回答当前问题的实验。

如果 overlap 很低，就基本坐实：当前 split-lite 的主要问题是 basis 方向错。
如果 overlap 不低，再去跑 `alpha/rank`，看是不是强度不够。
如果 prototype 结果不确定，再用旧任务 validation loss 做小规模确认。

---

### 2.12 v6 双环实现与分卡实验（2026-07-10）

v6 固定已验证的 `old_union top-16` 容量先验，并通过
`projection_scope=protected_only` 让它真正限定被保护 expert；`e_pk`、anchor、
replay 与分类器均不改动。

- **方向敏感保护（主机制）**：任务结束后，对历史 prototype 的 `e_pv` 输出构建
  top-16、rank-4 的功能 tangent basis。训练中仅对这 16 个 expert 投影；实时
  `conflict_e=||B_e g_e||^2 / ||g_e||^2` 决定增量保护，
  `alpha_e=0.2+0.3*(0.6*conflict_e+0.4*transient_risk_e)`，故基础保护不会被
  transient 降低。
- **瞬态 prompt（辅机制）**：仅 warm-up `e_pv` 20 batch，恢复权重前记录
  `Delta e`；用 2 个不更新 batch 计算新任务 signed gain，并以
  `||B_e Delta e||^2 / ||Delta e||^2` 计算旧功能风险。其兼容性只生成有界
  `0.1 * centered(cp)` router bias，并只贡献上式 40% 的增量保护。
- **可审计性**：`*_effective.log` 记录最终指标；`*_projection.log` 记录 basis
  覆盖率与逐 expert 的 conflict/alpha/risk；`*_transient.log` 记录 gain/risk/cp/
  router bias/delta norm。probe 会恢复参数、buffer、梯度、router 状态和 RNG。

不提供一键总控或自动筛选脚本；以下五个脚本相互独立，可按显卡分别启动。默认均为
`MAX_TASK=5 REPEAT=3 CRCT_EPOCHS=20 top-16 rank-4`，且各自写入不同目录：

```text
experiments/cifar-100_v6_exp1_legacy_topk16.sh       # GPU 0：gradient basis 对照
experiments/cifar-100_v6_exp2_functional_topk16.sh   # GPU 1：功能 basis，固定 alpha=0.2
experiments/cifar-100_v6_exp3_conflict_topk16.sh     # GPU 2：功能 basis + conflict 自适应
experiments/cifar-100_v6_exp4_transient_topk16.sh    # GPU 3：conflict + transient router bias
experiments/cifar-100_v6_exp5_dual_topk16.sh         # GPU 4：完整双环
```

例如四张卡并行时：

```bash
GPUID=0 bash experiments/cifar-100_v6_exp1_legacy_topk16.sh
GPUID=1 bash experiments/cifar-100_v6_exp2_functional_topk16.sh
GPUID=2 bash experiments/cifar-100_v6_exp3_conflict_topk16.sh
GPUID=3 bash experiments/cifar-100_v6_exp4_transient_topk16.sh
GPUID=4 bash experiments/cifar-100_v6_exp5_dual_topk16.sh
```

只有一至四张卡时，只需把命令最前的 `GPUID` 改成空闲卡号（例如把 Exp5 的
`GPUID=4` 改为 `GPUID=0`）；需要新一轮参数试验时，改用新的
`OUTDIR`，避免覆盖或混合旧日志。例如正式候选可运行：

```bash
GPUID=0 MAX_TASK=10 REPEAT=5 OUTDIR=outputs/cifar-100/10-task/v6-final-dual \
  bash experiments/cifar-100_v6_exp5_dual_topk16.sh
```

手工比对遵循严格配对：先确认各目录 `args.yaml` 的 `MAX_TASK`、`CRCT_EPOCHS`、
seed 与 `repeat` 相同，再直接对比 `effective.log` 中每个 repeat 的 FAA/CAA/FR。
比较链为 Exp1→Exp2（功能方向是否有效）、Exp2→Exp3（conflict 是否有增益）、
Exp3→Exp4（在 conflict 上加入 transient router 是否有益）、Exp3→Exp5（双环是否协同）。先看 Exp2
的 projection 覆盖率中位数是否不少于 0.20；满足后，只有 FAA 或 CAA 比匹配对照
提高至少 0.5 且 FR 不变差，才升级该候选至 10-task、5 seeds。不得把缺失的
repeat 或不同预算的目录混入比较。

其中最重要的是：

- `args.yaml`：确认实际参数
- `*_effective.log`：每个 repeat 的 FAA、CAA、FR
- `*_projection.log`：basis、projected vectors、conflict、alpha、removed ratio
- `*_transient.log`：gain、risk、router bias、delta norm
- `results-acc/*.yaml`：完整 accuracy history

除了 FAA/CAA/FR，还要检查：

- Exp2：`projection.log` 中是否有 functional basis、active experts、projected vectors
- Exp3：`mean_alpha` 是否经常高于基础值 `0.2`
- Exp4：`transient.log` 是否有非零 `gain`、`risk`、`router_bias`
- Exp5：是否同时出现 conflict 自适应 alpha 和 transient risk

如果 Exp3 的 alpha 始终等于 0.2，说明 conflict 没有触发；如果 Exp4 的 router bias 全为 0，说明 transient 信号没有产生作用。

### 2.13 `e_pv` 遗忘源因果审计（双配置、10-task）

`e_pk` 是 router key，决定一个样本选中哪些 expert；`e_pv` 是 value prefix，决定
选中 expert 注入注意力层的内容。`e_pv` 参数漂移表示其张量相对旧快照发生变化；
`e_pv` 功能漂移则是在旧类 input prototype 上，当前输出与保存输出的 MSE。后者是旧
功能变化的代理，不等于旧任务 accuracy：router 改变或 classifier/CRCT 改变也会造成遗忘。

下列两个脚本是严格配对的遗忘源审计。默认均为 `MAX_TASK=10`、`CRCT_EPOCHS=50`、
`REPEAT=3`、seeds `0 1 2`，各自必须使用独立输出目录：

```bash
GPUID=0 bash experiments/cifar-100_audit_no_projection.sh
GPUID=1 bash experiments/cifar-100_audit_functional_projection.sh
```

- `audit_no_projection`：保留既有 `e_pk/e_pv` anchor，使用 `--disable_split_lite`，不创建
  projector。
- `audit_functional_projection`：仅启用 `functional_tangent + protected_only + topk16 + rank4
  + alpha0.2`；不启用 conflict 或 transient。

两组都会写入 `causal_audit.jsonl`。任务 2–10 每次记录两个阶段：

- `post_main_pre_crct`：主训练结束、CRCT 前；
- `post_crct`：CRCT 后。

每条记录包含旧任务逐任务 accuracy、平均旧任务 accuracy、旧类 margin、`e_pv` 功能与
参数漂移、router logits MSE/KL/top-5 Jaccard。任务 5 和 10 还包含局部恢复反事实：临时
恢复任务 `t-1` 的 `e_pv`、`e_pk` 或 classifier head 后重新评估；
`restoration_accuracy_delta` 只表示该组件的局部贡献，三者不能相加。

手工比较先按同 seed 对齐，再看：

- projection 同时降低 `e_pv` 功能漂移、且在 `post_main_pre_crct` 提升旧任务 accuracy：
  `e_pv` 是有效干预点；
- T1 有收益、T2 消失：CRCT/head 覆盖了收益；
- 恢复 `e_pk` 的收益大于恢复 `e_pv`：router 更可能是遗忘源；
- 恢复 head 的收益主要出现在 T2：classifier/CRCT 更可能是遗忘源。

只有三个 seed 中方向一致，且最终 FAA 或 CAA 有实际提升、FR 不变差，才进入后续机制
优化；不要把机制日志本身当作性能提升证据。

---

## 3. 统一设计原则：Activation-Weighted Stability-Plasticity Trade-off

> v3 更新：每个参数子空间的保护强度与其在旧任务中的**激活频率**成正比。

| 参数类型 | 保护机制（v3） | 强度控制 |
|---------|---------------|---------|
| Expert Key (e_pk) | 权重空间 L2 正则 | 旧任务 expert 使用频率 → L2 权重 |
| Expert Value (e_pv) | 权重空间 L2 正则 + 特征蒸馏 | 旧任务 expert 使用频率 → L2 权重 |
| Prompt Features | e_pv 特征蒸馏（function space） | 旧任务各类 prototype 等权约束 |

---

## 4. SMoPE 数据流地图

```
                        ┌─────────────────────────────────┐
                        │        Input Batch (x)           │
                        │   来自当前任务 T_cur 的样本       │
                        └──────────────┬──────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────────┐
                        │      Frozen Pre-trained ViT      │
                        │  (标准 self-attention → A_pre)   │
                        └──────────────┬──────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             ▼                             │
         │              ┌──────────────────────────┐                 │
         │              │   Average Input Repr  x̃  │                 │
         │              └────────────┬─────────────┘                 │
         │                           │                               │
         │                           ▼                               │
         │              ┌──────────────────────────┐                 │
         │              │     Router / Gating       │  ◄── 组件一    │
         │              │  ┌────────────────────┐   │   KL 散度正则  │
         │              │  │ Score = f(x̃, K_expert)│   │   保护旧分布  │
         │              │  │ + Adaptive Noise     │   │              │
         │              │  │ → Top-K Selection    │   │              │
         │              │  └────────┬───────────┘   │              │
         │              └───────────┬───────────────┘              │
         │                          │ Top-K expert indices          │
         │                          ▼                               │
         │              ┌──────────────────────────┐                 │
         │              │   Prompt Expert 参数      │  ◄── 组件二    │
         │              │  ┌────────────────────┐   │  梯度投影到    │
         │              │  │ Expert 1: K₁, V₁   │   │  minor subspace│
         │              │  │ Expert 2: K₂, V₂   │   │ (频率加权)     │
         │              │  │ ... (sparse act.)   │   │              │
         │              │  │ Expert K: K_K, V_K │   │              │
         │              │  └────────┬───────────┘   │              │
         │              └───────────┬───────────────┘              │
         │                          │ Selected K_i, V_i             │
         │                          ▼                               │
         │              ┌──────────────────────────┐                 │
         │              │  Ã_prompt (prompt attn)  │                 │
         │              │  = RowExpand(Scores)     │                 │
         │              └────────────┬─────────────┘                 │
         │                           │                               │
         └───────────────────────────┼───────────────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Final Attention Matrix      │
                      │  A_final = [A_pre | Ã_prompt]│
                      └────────────┬────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────────┐
                      │  Dot Product with Expert     │
                      │  Representations → Output    │
                      └────────────┬────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────────┐
                      │     Classification Head      │
                      │      → CE Loss (L_ce)        │
                      └────────────┬────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         ▼                         │
         │  ┌──────────────────────────────────────────────┐ │
         │  │          Loss Aggregation (每 batch)          │ │
         │  │                                              │ │
         │  │  L_total = L_ce                              │ │
         │  │           + λ_router · L_KL          (组件一) │ │
         │  │           + λ_key · L_key_rel        (组件三) │ │
         │  │           + λ_proto · L_proto_align  (组件三) │ │
         │  │                                              │ │
         │  │  其中 L_ce 的梯度在反传时经过                   │ │
         │  │  SplitLoRA 式 minor-subspace 投影 (组件二)     │ │
         │  └──────────────────────┬───────────────────────┘ │
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────────┐
                      │   Alternating Update 策略    │
                      │  Step 1: Update Expert/     │
                      │          Prompt/Router       │
                      │          (key 冻结)           │
                      │  Step 2: Update Key/         │
                      │          Prototype           │
                      │          (其他冻结)           │
                      └─────────────────────────────┘
```

---

## 5. 结构化伪代码

### 5.1 全局数据结构

```python
# ============================================================
# GLOBAL STATE (跨任务持久化)
# ============================================================

class TaskMemory:
    """每完成一个任务后保存的关键信息"""
    task_id: int
    num_classes: int

    # --- 组件一：Router 分布约束 ---
    router_prototypes: Tensor        # [num_classes, d_router]
                                     # 每个类的 router logits 均值向量

    # --- 组件二：Expert 梯度投影 ---
    global_major_subspace: Tensor    # [d_total, r]
                                     # 所有 expert 参数的全局 major subspace
    expert_usage_freq: Tensor        # [K]
                                     # 每个 expert 在旧任务中的激活频率

    # --- 组件三：Key 几何稳定性 ---
    key_prototypes: Tensor           # [num_classes, d_key]
                                     # 每个类的 key prototype 矩阵 K_t
    key_pairwise_sim: Tensor         # [num_classes, num_classes]
                                     # 旧 key 的 pairwise 相似度矩阵 S_t = K_t K_t^T
```

### 5.2 任务完成时：后处理

```python
def on_task_finish(task_id: int, model: SMoPE, dataloader: DataLoader):
    """
    在每个任务训练完成后调用。
    估计旧任务的梯度空间、保存分布/几何约束所需的信息。
    """
    memory = TaskMemory(task_id=task_id, num_classes=dataloader.num_classes)

    # ── 组件一：保存 Router Prototype Distribution ──
    all_router_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            logits = model.router.get_logits(x)       # [B, K]
            all_router_logits.append(logits)
            all_labels.append(y)
    all_router_logits = torch.cat(all_router_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Per-class mean of router logits
    for c in range(memory.num_classes):
        mask = (all_labels == c)
        memory.router_prototypes[c] = all_router_logits[mask].mean(dim=0)

    # ── 组件二：估计 Global Major Subspace ──
    # 收集所有 expert/prompt 参数的梯度（拼接后统一 SVD）
    all_grads = []
    for x, y in dataloader:
        model.zero_grad()
        loss = model.compute_ce_loss(x, y)
        loss.backward()
        # 拼接所有 expert MLP + prompt key/value 的梯度
        grad_vec = collect_expert_gradients(model)    # [d_total]
        all_grads.append(grad_vec)
    grad_matrix = torch.stack(all_grads, dim=0)       # [N_samples, d_total]

    # SVD 取 major subspace（保留前 r 维 = 解释 95% 方差的方向）
    U, S, Vh = torch.linalg.svd(grad_matrix.float(), full_matrices=False)
    explained_var = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
    r = torch.searchsorted(explained_var, 0.95).item() + 1
    memory.global_major_subspace = Vh[:r, :].T          # [d_total, r]

    # ── Expert 使用频率 ──
    usage_counts = model.router.get_expert_usage_counts(dataloader)
    memory.expert_usage_freq = usage_counts / usage_counts.sum()

    # ── 组件三：Key Prototypes & Pairwise Similarity ──
    all_keys = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            keys = model.get_prompt_keys(x)             # [B, d_key]
            all_keys.append(keys)
            all_labels.append(y)
    all_keys = torch.cat(all_keys, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    for c in range(memory.num_classes):
        mask = (all_labels == c)
        memory.key_prototypes[c] = all_keys[mask].mean(dim=0)

    memory.key_pairwise_sim = memory.key_prototypes @ memory.key_prototypes.T
    # S_t = K_t K_t^T, shape [C_t, C_t]

    return memory
```

### 5.3 新任务训练：核心循环

```python
def train_new_task(
    model: SMoPE,
    dataloader: DataLoader,
    old_memories: List[TaskMemory],
    hyperparams: Dict,
):
    """
    在新任务上训练 SMoPE，同时施加异构保护约束。
    """
    optimizer_ce = AdamW(model.expert_and_prompt_params(), lr=hyperparams['lr'])
    optimizer_key = AdamW(model.key_params(), lr=hyperparams['lr_key'])

    for epoch in range(hyperparams['epochs']):
        for x, y in dataloader:

            # ═══════════════════════════════════════════════
            # Step 1: CE Loss + 前向 + 梯度投影 (组件二)
            # ═══════════════════════════════════════════════
            model.train()
            model.freeze_keys()   # key 冻结
            optimizer_ce.zero_grad()

            logits = model(x)
            L_ce = F.cross_entropy(logits, y)
            L_total = L_ce

            # ── 组件一：Router KL 散度正则 ──
            L_kl = compute_router_kl_loss(model, old_memories)
            L_total = L_total + hyperparams['lambda_router'] * L_kl

            # ── 组件三前半：Prototype Alignment ──
            L_proto = compute_prototype_alignment(model, old_memories)
            L_total = L_total + hyperparams['lambda_proto'] * L_proto

            L_total.backward()

            # ── 组件二：梯度投影到 Minor Subspace ──
            project_gradients_to_minor_subspace(
                model, old_memories
            )

            optimizer_ce.step()

            # ═══════════════════════════════════════════════
            # Step 2: Key Relation Distillation (组件三后半)
            # ═══════════════════════════════════════════════
            model.unfreeze_keys()
            model.freeze_experts()  # expert/prompt/router 冻结
            optimizer_key.zero_grad()

            L_key_rel = compute_key_relation_loss(model, old_memories)
            L_key_rel.backward()
            optimizer_key.step()

            model.unfreeze_experts()
```

### 5.4 组件一：Router KL 散度正则（详细实现）

```python
def compute_router_kl_loss(model: SMoPE, old_memories: List[TaskMemory]) -> Tensor:
    """
    对每个旧任务的每个类，约束 router 在当前参数下对
    "该类 prototype 输入" 的输出分布不偏离旧分布太远。

    L_KL = Σ_t Σ_c KL( P_old(router|x̄_t,c) || P_cur(router|x̄_t,c) )
    """
    if not old_memories:
        return torch.tensor(0.0, device=model.device)

    total_kl = 0.0
    for mem in old_memories:
        for c in range(mem.num_classes):
            # 旧分布：保存在 mem 中的 router logits prototype
            # 用 softmax 转为概率分布
            old_logits = mem.router_prototypes[c].to(model.device)   # [K]
            old_probs = F.softmax(old_logits, dim=-1)

            # 当前分布：用当前 router 参数，输入该类 prototype 对应的
            # 平均输入表征（也保存在 mem 中或通过 key prototype 反推）
            # 简化：直接用 router 对 key_prototype 的输出
            cur_logits = model.router(mem.key_prototypes[c])         # [K]
            cur_probs = F.softmax(cur_logits, dim=-1)

            # KL(P_old || P_cur)
            kl = (old_probs * (old_probs.log() - cur_probs.log())).sum()
            total_kl += kl

    return total_kl / len(old_memories)
```

### 5.5 组件二：梯度投影（详细实现）

```python
def project_gradients_to_minor_subspace(
    model: SMoPE, old_memories: List[TaskMemory]
):
    """
    对每个 expert 的梯度，将其在旧任务 major subspace 上的分量削弱/剔除，
    只保留 minor subspace 中的分量。保护强度由 expert 使用频率决定。

    对 expert i：
        g_i ← g_i - α_i · P_major(g_i)
    其中：
        P_major(g_i) = U U^T g_i（投影到 major subspace）
        α_i = min(1, freq_i / freq_threshold)（频率越高的 expert 保护越强）
    """
    if not old_memories:
        return

    # 聚合所有旧任务的 global major subspace（取平均或拼接后重做 SVD）
    # 简化：使用最新旧任务的 major subspace
    U = old_memories[-1].global_major_subspace   # [d_total, r]

    for i, expert in enumerate(model.prompt_experts):
        # 收集该 expert 所有参数的梯度
        grads = []
        for p in expert.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if not grads:
            continue
        g = torch.cat(grads)                     # [d_i]

        # 计算 protection strength α_i
        freq = max(mem.expert_usage_freq[i] for mem in old_memories)
        alpha = min(1.0, freq / FREQ_THRESHOLD)  # alpha ∈ [0, 1]

        if alpha > 0:
            # 投影到 major subspace
            g_major = U[:len(g)] @ (U[:len(g)].T @ g)
            # 削弱 major 方向上的分量
            g_projected = g - alpha * g_major

            # 回写到各参数的 .grad
            offset = 0
            for p in expert.parameters():
                if p.grad is not None:
                    n = p.grad.numel()
                    p.grad.copy_(g_projected[offset:offset + n].view_as(p.grad))
                    offset += n
```

### 5.6 组件三：Key Relation Distillation Loss（详细实现）

```python
def compute_key_relation_loss(
    model: SMoPE, old_memories: List[TaskMemory]
) -> Tensor:
    """
    约束旧任务 key 之间的 pairwise 相似度结构不被破坏。

    L_key_rel = Σ_t || S_t - Ŝ_t ||_F²
    其中：
        S_t = K_t K_t^T（旧 key prototype 的相似度矩阵）
        Ŝ_t = K̂_t K̂_t^T（当前参数下的 key prototype 相似度矩阵）
    """
    if not old_memories:
        return torch.tensor(0.0, device=model.device)

    total_loss = 0.0
    for mem in old_memories:
        # 当前 key prototypes
        cur_keys = model.get_key_prototypes_for_task(mem.task_id)  # [C_t, d_key]

        # 当前 pairwise 相似度
        cur_sim = cur_keys @ cur_keys.T                           # [C_t, C_t]

        # 旧 pairwise 相似度（已保存）
        old_sim = mem.key_pairwise_sim.to(model.device)            # [C_t, C_t]

        # Frobenius 范数
        total_loss += F.mse_loss(cur_sim, old_sim)

    return total_loss / len(old_memories)


def compute_prototype_alignment(model: SMoPE, old_memories: List[TaskMemory]) -> Tensor:
    """
    可选的 prototype alignment loss：
    约束当前 key prototype 不远离旧 key prototype 的绝对位置。
    """
    if not old_memories:
        return torch.tensor(0.0, device=model.device)

    total_loss = 0.0
    for mem in old_memories:
        cur_keys = model.get_key_prototypes_for_task(mem.task_id)
        old_keys = mem.key_prototypes.to(model.device)
        total_loss += F.mse_loss(cur_keys, old_keys)

    return total_loss / len(old_memories)
```

---

## 6. 超参数配置

| 超参数 | 建议范围 | 含义 |
|--------|---------|------|
| `λ_router` | 0.01 ~ 0.5 | Router KL 散度正则的权重 |
| `λ_key` | 0.1 ~ 1.0 | Key Relation Distillation Loss 的权重 |
| `λ_proto` | 0.01 ~ 0.1 | Prototype Alignment Loss 的权重 |
| `FREQ_THRESHOLD` | 1/K ~ 3/K | Expert 使用频率阈值，低于此值 α=0（不保护），高于此值 α 线性增长 |
| `r` (major subspace dim) | explained_var ≥ 0.95 | SVD 保留的 major subspace 维度数 |
| `lr` (expert/prompt) | 1e-3 ~ 1e-4 | Expert 和 prompt 参数学习率 |
| `lr_key` | 1e-4 ~ 1e-5 | Key 参数学习率（通常设得更小以保持稳定） |

---

## 7. 消融实验设计（Ablation Study）

### 7.1 核心消融：证明异构约束的必要性

| 实验 | Router | Expert | Key | 预期 |
|------|--------|--------|-----|------|
| A (No protection) | 无约束 | 无投影 | 无约束 | 塑性好，稳定性差 |
| B (Uniform SplitLoRA) | 硬投影 | 硬投影 | 硬投影 | 稳定好，塑性差 |
| C (Ours - Router) | KL 散度 | 无投影 | 无约束 | — |
| D (Ours - Expert) | 无约束 | 梯度投影 | 无约束 | — |
| E (Ours - Key) | 无约束 | 无投影 | Relation Distill | — |
| F (Ours - Full) | KL 散度 | 梯度投影 | Relation Distill | **最优平衡** |

### 7.2 附加消融

| 实验 | 变量 | 目的 |
|------|------|------|
| G | Shared vs Per-Expert Subspace | 验证 Shared Subspace 是否足够 |
| H | Frequency-weighted vs Uniform α | 验证频率加权自适应的价值 |
| I | Alternating vs Joint Update | 验证交替更新策略的必要性 |
| J | KL vs L2 vs Cosine for Router | 验证 KL 散度的选择 |

---

## 8. 实现路线图

```
Phase 1: 复现 Baseline（SMoPE 原论文）
  ├── 跑通 SMoPE 原始代码
  ├── 在 2~3 个 CL benchmark 上复现结果
  └── 确认数据流和关键模块位置

Phase 2: 实现组件二（Expert 梯度投影）
  ├── 实现 on_task_finish() 中的梯度收集 + SVD
  ├── 实现 project_gradients_to_minor_subspace()
  └── 先做 uniform α（不做频率加权），验证基础投影有效

Phase 3: 实现组件三（Key 几何约束）
  ├── 实现 Key Relation Distillation Loss
  ├── 实现 Alternating Update 策略
  └── 验证 key 约束单独有效

Phase 4: 实现组件一（Router KL 正则）
  ├── 实现 Router Prototype 保存
  ├── 实现 KL 散度正则项
  └── 验证 router 约束单独有效

Phase 5: 联合调优 + 消融实验
  ├── 调 λ_router, λ_key, λ_proto, FREQ_THRESHOLD
  ├── 加频率加权自适应 α
  ├── 跑完整消融实验矩阵
  └── 收集最终结果

Phase 6: 论文写作
  ├── 撰写方法部分
  ├── 绘制架构图
  └── 完成实验分析
```

---

## 9. 风险 & 注意事项

| 风险 | 等级 | 应对 |
|------|------|------|
| SVD 在大 d_total 下计算开销大 | 中 | 使用随机 SVD (randomized SVD) 近似；或按 layer 分组独立做 SVD |
| Alternating update 导致训练慢 2× | 低 | 可每 N 个 batch 做一次 key update，而非每个 batch |
| Key Relation Loss 在大 C_t 时 O(C_t²·d) | 低 | C_t 通常不大（每任务类别数有限）；可做采样近似 |
| Router KL 需要旧类 prototype 的输入表征 | 中 | 保存每个类的平均输入 x̄，或直接用 key_prototype 作为 proxy |
| 多个旧任务时约束项累加过多 | 中 | 随机采样旧任务子集；或用 memory bank 做 replay-based 近似 |

---

## 10. 文件结构（推荐）

```
SMOPE/
├── guide.md                          # ← 本文件（项目核心指南）
├── src/
│   ├── model/
│   │   ├── smope.py                  # SMoPE 原始模型
│   │   ├── router.py                 # Router / Gating 模块
│   │   ├── prompt_expert.py          # Prompt Expert 模块
│   │   └── key_prototype.py          # Key / Prototype 模块
│   ├── protection/
│   │   ├── task_memory.py            # TaskMemory 数据结构
│   │   ├── router_kl.py              # 组件一：Router KL 散度正则
│   │   ├── gradient_projection.py    # 组件二：梯度投影
│   │   └── key_relation.py           # 组件三：Key Relation Distillation
│   ├── training/
│   │   ├── train_task.py             # 新任务训练循环
│   │   ├── on_task_finish.py         # 任务完成后处理
│   │   └── alternating_update.py     # Alternating Update 策略
│   └── utils/
│       ├── svd_utils.py              # SVD 工具（含 randomized SVD）
│       └── metrics.py                # CL 评估指标 (FM, PL, ACC)
├── configs/
│   └── default.yaml                  # 默认超参数配置
├── experiments/
│   └── ablation.md                   # 消融实验记录
└── README.md
```

---

> **最后更新**：2026-06-26（v4-split-lite 在线低秩软投影与有效数据日志完成）
> **下次对话**：读取本文件即可恢复全部上下文，无需重复描述项目背景。
> **当前版本**：v4-split-lite — 轻量 router 稳态 + e_pv anchor + SplitLoRA-style 在线低秩软投影；默认启用。
> **分支**：`v4`
