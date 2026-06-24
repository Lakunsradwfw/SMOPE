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

## 3. 统一设计原则：Activation-Weighted Stability-Plasticity Trade-off

> 每个参数子空间（router / expert / key）的保护强度与其在旧任务中的**激活频率**成正比。

| 参数类型 | 保护机制 | 强度控制 |
|---------|---------|---------|
| Router logits | KL 散度正则 | 旧类 prototype 的置信度加权 |
| Expert MLP / Prompt | 梯度投影到 minor subspace | 旧任务 expert 使用频率 → projection strength |
| Key / Prototype | Relation Distillation Loss | 旧任务类间距离的稳定性加权 |

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

> **最后更新**：2025-07-15（v2 优化完成）
> **下次对话**：读取本文件即可恢复全部上下文，无需重复描述项目背景。
> **当前版本**：v2 — 诊断日志 + NaN 修复 + 增量 SVD + 温度软化；默认启用，超参数已优化。
