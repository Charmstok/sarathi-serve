# Active Prefill Control (APC)

上一小节的时间预算机制主要回答的是“在给定系统状态下，本轮 batch 允许扩张到何种 token 规模”这一问题，其核心作用是通过 target-time 约束控制单轮调度的时间开销。然而，仅有时间预算控制仍不足以解决混合 `decode/prefill` 场景中的结构性失衡：当 `decode` 长时间占据主导时，waiting 队列中的 prefill 可能持续得不到进入机会；而当 unfinished prefill 无约束增长时，系统又容易产生大量碎片化的小 chunk 推进。基于此，本节进一步从 **unfinished prefill 活跃度控制** 的角度补充时间预算机制，提出一种与 target-time 调度互补的调度增强方法，即 **Active Prefill Control (APC)**。若将时间预算机制理解为“控制 batch 扩张的 token 维度约束”，则 APC 可以理解为“控制 unfinished prefill 扩张的结构维度约束”。二者并非替代关系，而是分别作用于调度器的两个正交维度：前者决定“本轮最多还能放多少 token”，后者决定“哪些 unfinished prefill 值得在本轮保持活跃”。

## 1. 问题定义

考虑一个同时处理 `decode` 与 `prefill` 的迭代式批处理调度器。在第 $t$ 轮调度中，系统需要从当前运行集合与等待队列中构造一个 batch，用于执行下一轮 GPU 计算。记该轮 batch 为：

$$
\mathcal{B}_t = \mathcal{D}_t \cup \mathcal{P}_t,
$$

其中，$\mathcal{D}_t$ 表示本轮被调度的 `decode` 序列集合，$\mathcal{P}_t$ 表示本轮被调度的 active unfinished prefill 序列集合。系统至少受到以下两类硬约束限制。

首先是序列槽位约束：

$$
|\mathcal{B}_t| \le S_{\max},
$$

其中 $S_{\max}$ 为系统允许的最大并发序列数。其次是 token 预算约束：

$$
\sum_{j \in \mathcal{B}_t} \mathrm{tokens}(j) \le T_{\max},
$$

其中 $T_{\max}$ 表示当前轮调度允许的最大 token 预算，该预算由上一小节的时间预算机制隐式决定。

在该场景中，调度器面临两个相互竞争的优化目标。其一，已经进入解码阶段的请求需要获得尽可能连续的 `decode` 推进，否则将直接增加在服请求的尾部等待时间。其二，waiting 队列中的新请求也需要及时获得 `prefill` 机会，否则系统会逐渐退化为长期纯 `decode` 状态，导致新请求的首 token 延迟与整体排队时间显著升高。问题在于，这两类目标往往不能同时无代价满足。若允许大量 unfinished prefill 同时活跃，则每条 prefill 在单轮中往往只能得到很小的 chunk，形成碎片化推进；若过度偏向 `decode`，则 prefill 又可能长期饥饿。因此，本文关注的问题不是“是否允许 prefill 进入 batch”，而是：

> 在既有时间预算约束下，如何以一种受控方式维持 unfinished prefill 的最小必要活性，从而在不过度破坏 decode 连续性的前提下改善新请求的 prefill 推进质量？

换言之，本文试图求解的是一个结构层面的调度平衡问题：在有限 seq slot 与 token budget 约束下，系统究竟应当允许多少 unfinished prefill 同时保持活跃，以及每条 active unfinished prefill 至少应推进到何种程度，才可以认为这一轮调度是“值得的”。

## 2. 方法设计

### 2.1 设计动机

本文提出 **Active Prefill Control (APC)**，其核心思想是：将 unfinished prefill 视为一种需要显式控制的活跃资源，而不是默认允许其无约束地附着在当前 batch 中。APC 的设计目标并非静态为 prefill 预留固定资源，而是在每一轮调度时根据当前 batch 结构动态决定：

1. 当前最多允许多少 unfinished prefill 保持活跃；
2. 每条 active unfinished prefill 至少应获得多少 token 推进；
3. 当系统退化为纯 `decode` 运行时，如何以最低代价重新恢复 prefill 活性。

因此，APC 的本质是一种 **动态 unfinished prefill 活跃度控制机制**。它不改变原有 target-time 调度器对 chunk 大小的评分逻辑，而是在其基础上附加一个结构层面的筛选条件，从而减少低效的 prefill 碎片化推进，并避免系统长期停留在 prefill 完全失活的状态。

### 2.2 动态活跃度上界

定义两个核心控制参数：

$$
C_{\max} = \texttt{max\_active\_prefill\_seqs}, \qquad
L_{\min} = \texttt{min\_active\_prefill\_chunk\_size}.
$$

其中，$C_{\max}$ 表示允许同时处于 active 状态的 unfinished prefill 序列上限，$L_{\min}$ 表示单条 active unfinished prefill 在一轮中应获得的最小有效 chunk 大小。

在第 $t$ 轮调度时，APC 根据当前 decode 占用与剩余 token 预算，计算 unfinished prefill 的动态允许上界：

$$
C_t = \min \left(
C_{\max},
S_{\max} - |\mathcal{D}_t|,
\left\lfloor \frac{T_{\max} - N_t}{L_{\min}} \right\rfloor
\right),
$$

其中，$N_t$ 表示在处理 unfinished prefill 之前，当前 batch 已经承诺给其他序列的 token 数。上述公式具有三个层面的含义。

第一，unfinished prefill 的活跃数量不应超过显式配置上限 $C_{\max}$，否则会造成过度并发。第二，unfinished prefill 的活跃数量不能超过剩余 seq slot 数，即 `decode` 已经占据的槽位会直接压缩 prefill 的活跃空间。第三，即使 seq slot 仍然足够，若剩余 token 预算已经不足以为每条 active unfinished prefill 提供至少 $L_{\min}$ 个 token，则这些 unfinished prefill 也不应在本轮同时活跃。由此可见，APC 控制的并不是单一意义上的“活跃序列数”，而是“在当前系统状态下，既满足结构约束又满足最小推进约束的 unfinished prefill 活跃数”。

### 2.3 最小有效推进约束

对任意 unfinished prefill 序列 $i$，记其剩余 prompt token 数为 $r_i$，则定义该序列在当前阶段的最小有效推进粒度为：

$$
m_i = \min(r_i, L_{\min}).
$$

该定义意味着：若该序列本身剩余 prompt 已不足 $L_{\min}$，则允许其以剩余长度作为本轮有效推进；否则，只有当本轮至少能为其分配到 $L_{\min}$ 个 token 时，才认为该轮推进是有意义的。该约束的目的在于抑制大量极小 prefill chunk 的出现。因为在高压混部场景下，unfinished prefill 往往会被切成非常碎片化的小块，这类推进虽然形式上增加了 prefill 的调度次数，但实际上很难带来实质性的 prompt 清空收益，反而会破坏 batch 稳定性。

### 2.4 与原时间预算搜索的耦合方式

APC 不替换原有基于时间预算的 chunk 搜索，而是在原有搜索结果上再增加一层可接受性判断。设原调度器为序列 $i$ 给出的候选 chunk 为：

$$
u_i^\star = \arg\min_{1 \le u \le h_i} \phi_t(u),
$$

其中，$h_i$ 表示当前 batch 状态下序列 $i$ 可行的最大 chunk 上界，$\phi_t(u)$ 表示由 target-time 机制给出的评分函数。APC 不改变 $\phi_t(u)$ 的定义，而是在候选 chunk 得到之后，再判断该 chunk 是否满足 active unfinished prefill 的结构约束。具体而言：

$$
u_i =
\begin{cases}
\nu_i^\star, & \text{若 } |\mathcal{P}_t| < C_t \text{ 且 } \nu_i^\star \ge m_i, \\
\min(h_i, m_i), & \text{若当前 batch 中尚无 active prefill，触发 warm-start}, \\
\varnothing, & \text{其他情况。}
\end{cases}
$$

上述规则表明，APC 的作用不是决定“最优 chunk 大小”，而是决定“该 unfinished prefill 是否值得在当前轮次保持活跃”。若其候选 chunk 已经满足最小有效推进要求，则直接沿用原时间预算搜索结果；若不满足，则默认认为该 prefill 在当前状态下仅会带来低效碎片化推进，因此不应保留在 active 集合中。只有当当前 batch 中已经不存在任何 active prefill 时，系统才允许进行一次低代价的强制启动，以避免调度状态彻底坍缩为纯 `decode` 批次。

### 2.5 Warm-start 机制

warm-start 是 APC 中保证 prefill 活性不会归零的关键补丁。若系统在若干轮调度后进入了

$$
|\mathcal{P}_t| = 0
$$

的状态，则后续 batch 很可能完全由 `decode` 组成。此时，若仍然要求 waiting prefill 必须先满足严格的最小 chunk 条件，系统就可能长期停留在“没有 prefill 参与，因此也很难重新拉起 prefill”的坏平衡点。为打破这一退化状态，APC 引入如下 warm-start 规则：当且仅当当前 batch 中没有 active unfinished prefill 时，允许对一条候选 prefill 以

$$
\min(h_i, m_i)
$$

作为启动 chunk 强制进入 batch。需要强调的是，warm-start 并不意味着无限制地偏向 prefill；它只是在 prefill 完全失活时为系统恢复 prefill 活性提供一条最小干预路径。因此，APC 的目标不是让更多 prefill 永久活跃，而是让系统始终维持一个“非零但可控”的 unfinished prefill 活性水平。

## 3. 算法流程

### 3.1 调度流程描述

APC 的在线调度流程可分为三个阶段。

第一阶段，优先处理 `decode` 序列。对于已完成 prompt 阶段的请求，调度器优先为其分配下一轮解码 token，并沿用原有的显存检查与必要的 preemption 逻辑。该阶段结束后，系统可以得到当前 decode 占用状态 $|\mathcal{D}_t|$ 以及已承诺 token 数 $N_t$。

第二阶段，处理当前 running 集合中的 unfinished prefill。对每条该类序列 $i$，系统首先基于当前 decode 占用与 token 预算计算 active unfinished prefill 上界 $C_t$。若当前 active unfinished prefill 数已经达到 $C_t$，则该序列不会继续保留在 active 集合中，而是被延后回 waiting 队列。若尚未达到上界，则继续基于时间预算搜索得到候选 chunk $\nu_i^\star$。当 $\nu_i^\star \ge m_i$ 时，该序列被接纳进入本轮 batch；否则，若当前 batch 中尚无 active unfinished prefill，则触发 warm-start，以 $\min(h_i, m_i)$ 作为启动 chunk；若仍不满足条件，则将其 defer 回 waiting 队列。

这里的 defer 只改变调度状态，不回退已经完成的 prompt 进度。记序列 $i$ 在第 $t$ 轮结束时已经完成的 prompt token 数为 $p_i^{(t)}$，则 defer 满足：

$$
p_i^{(t+1)} = p_i^{(t)}.
$$

因此，APC 不会通过回滚 prefill 进度来重构 batch，而只是将“当前不值得继续活跃”的 unfinished prefill 暂时移出 active 集合。

第三阶段，处理 waiting 队列中的新 prefill 请求。其逻辑与第二阶段基本一致：先检查显存条件与最大槽位限制，再计算当前轮 unfinished prefill 上界 $C_t$，最后根据 target-time 搜索结果与最小有效推进约束决定是否接纳。若满足条件，则为该请求正式分配显存并加入 running 集合；若不满足，则继续保留在 waiting 队列中等待下一轮评估。

因此，APC 在任意轮调度中始终维持以下不变量：

$$
|\mathcal{D}_t| + |\mathcal{P}_t| \le S_{\max},
$$

$$
\sum_{j \in \mathcal{B}_t} \mathrm{tokens}(j) \le T_{\max},
$$

$$
|\mathcal{P}_t| \le C_t.
$$

从实现层面看，APC 并不是一个完全替代原调度器的新系统，而是在原有时间预算调度框架上增加一层 unfinished prefill 活跃度控制模块。换言之，原有 target-time 机制继续回答“在可调度时给多大 chunk”，而 APC 回答“哪些 unfinished prefill 值得在本轮保持活跃”。

### 3.2 Algorithm 1: Active Prefill Control

```text
Algorithm 1 Active Prefill Control (APC)
Input:
  Running set R, waiting queue W, current time t
  Decode budget T_max, seq-slot budget S_max
  APC parameters C_max and L_min
Output:
  Scheduled batch B_t

1:  Initialize B_t <- empty, P_t <- empty
2:  Sort running sequences by the base scheduler priority
3:  Schedule decode sequences first and update:
      |D_t|, N_t, current running batch state
4:  Compute active-prefill cap:
      C_t <- min(C_max, S_max - |D_t|, floor((T_max - N_t) / L_min))
5:  for each unfinished prefill i in running set do
6:      if |P_t| >= C_t then
7:          defer i back to waiting queue
8:          continue
9:      end if
10:     obtain candidate chunk nu_i^* from the base target-time search
11:     set m_i <- min(r_i, L_min)
12:     if nu_i^* >= m_i then
13:         admit i with chunk nu_i^*
14:         add i into P_t and B_t
15:     else if |P_t| = 0 then
16:         warm-start i with chunk min(h_i, m_i)
17:         add i into P_t and B_t
18:     else
19:         defer i back to waiting queue
20:     end if
21: end for
22: for each waiting prefill i in queue order do
23:     if memory or seq-slot constraint is violated then
24:         break
25:     end if
26:     recompute C_t under current batch state
27:     if |P_t| >= C_t then
28:         break
29:     end if
30:     obtain candidate chunk nu_i^* from the base target-time search
31:     set m_i <- min(r_i, L_min)
32:     if nu_i^* >= m_i then
33:         admit i with chunk nu_i^*
34:         add i into P_t and B_t
35:     else if |P_t| = 0 then
36:         warm-start i with chunk min(h_i, m_i)
37:         add i into P_t and B_t
38:     else
39:         keep i in waiting queue and stop admission
40:     end if
41: end for
42: return B_t
```

该伪代码清晰反映了 APC 与原时间预算调度器之间的职责分工。步骤 3 和步骤 10/30 中的 chunk 搜索仍然由原有 target-time 机制完成；而步骤 4、11、15、27 和 35 则构成 APC 的附加控制逻辑。也就是说，APC 并不试图重新定义时间预算搜索器，而是通过 active unfinished prefill 上界、最小有效推进约束和 warm-start 机制，对原调度过程施加结构层面的稀疏化与稳定化控制。

## 4. 实验假设与设计思路

APC 的实验目标不是单纯追求吞吐最大化，而是验证其是否能够在 `decode` 主导的混合负载下，以有限的 decode 干扰代价改善 prefill 的进入与推进质量。因此，本文围绕以下三个核心假设展开实验设计。

**假设一：** 当负载显著偏向 `decode` 主导时，基线调度器更容易形成大规模 `decode` 占用，waiting prefill 难以及时进入 batch。在这一场景下，APC 通过显式限制 unfinished prefill 的活跃并发度，并维持非零 prefill 活性，应能够降低 `prefill_e2e_time`，尤其是在请求到达较为聚簇、decode 压力较大的区间中体现更明显。

**假设二：** 当 unfinished prefill 数量过多时，系统通常会出现大量小 chunk 推进，从而导致单条 prefill 的有效推进不足。在这一场景下，APC 应当提高平均 prefill chunk 大小，并减少碎片化 prefill 的比例。为验证这一点，实验需要重点观察 `avg_prefill_chunk`、`scheduled_prefill_seq_count`，以及 waiting prefill 因 active 上界或最小 chunk 约束被阻塞的次数。

**假设三：** APC 对 `decode` 的局部连续性可能带来一定扰动，因此其收益未必总能同步反映在整体 `request_e2e_time` 上。换言之，APC 更可能首先改善新请求的进入与 prompt 推进质量，而其对请求总完成时间的影响还取决于数据分布、target-time 配置以及当前 decode 压力水平。因此，实验不能只报告单一终端指标，而应同时观测收益指标与系统代价指标。

基于上述假设，实验评估建议至少覆盖以下四类指标。

第一类为 **请求级性能指标**，包括 `prefill_e2e_time`、`request_e2e_time` 的均值、P90 及其他尾部统计，用于评估 APC 对请求体验的直接影响。第二类为 **批次级行为指标**，包括 `decode-only batch` 的比例、平均 `scheduled_prefill_seq_count`、平均 `avg_prefill_chunk`，用于刻画 APC 对 batch 结构的实际干预效果。第三类为 **控制行为指标**，包括 `active_prefill_seq_count`、`deferred_prefill_seq_count`、`waiting_prefill_blocked_by_cap` 与 `waiting_prefill_blocked_by_min_chunk`，用于验证 APC 是否真正触发了设计中的控制路径。第四类为 **系统代价指标**，包括 batch latency、decode token 推进速度等，用于刻画 APC 改善 prefill 推进所付出的潜在代价。

在负载构造方面，为了尽可能放大 APC 的可观测效应，实验应优先采用 `decode` 偏重的数据分布。例如，可以使用“短 prompt + 长 decode”为主、少量“长 prompt + 短 decode”为辅的异构请求集合，并通过聚簇式到达模式强化同一时间窗口内的资源竞争。在参数设置方面，$C_{\max}$ 不宜过大，否则 unfinished prefill 控制会退化为接近无约束放行；$L_{\min}$ 也不宜过大，否则系统会频繁拒绝 prefill 进入，反而造成饥饿。因此，实验中应将 $C_{\max}$ 与 $L_{\min}$ 视为两个独立控制旋钮，系统性比较不同参数组合下 APC 的收益区间与退化区间。

进一步地，从方法定位上看，APC 并不追求在所有负载下都无条件优于基线。更合理的实验目标是回答以下两个问题：第一，在何种负载结构下，unfinished prefill 活跃度控制能够稳定转化为 prefill 侧收益；第二，在何种参数区间内，这种收益不会被 decode 侧代价所抵消。只有在这两个问题都得到定量回答之后，APC 才能被视为一种可解释、可部署、且具备明确适用边界的调度增强机制。

## 5. 流程图与机制图文字版说明

### 5.1 流程图文字说明

该流程图建议采用自上而下的布局，共分为六个模块，分别对应 APC 在一次调度迭代中的主要决策阶段。图的起点为“输入当前调度状态”，其输入包括 running 集合、waiting 队列、当前 decode 占用、剩余 token 预算以及 APC 参数 $(C_{\max}, L_{\min})$。从该模块向下连接至“优先调度 decode 序列”模块，用于表示 APC 并不改变基线调度器对 decode 的优先处理顺序，而是首先继承当前 batch 中的 decode 状态，并据此更新本轮已经承诺的 token 数与剩余 seq slot。

在“优先调度 decode 序列”之后，流程进入“计算 active unfinished prefill 上界”模块。该模块内部可直接标注公式

$$
C_t = \min \left(C_{\max}, S_{\max} - |\mathcal{D}_t|, \left\lfloor \frac{T_{\max} - N_t}{L_{\min}} \right\rfloor \right),
$$

用于表示 APC 在每轮调度中并非固定允许某个数量的 prefill 活跃，而是根据当前 decode 占用和剩余 token 预算动态决定 unfinished prefill 的最大活跃数量。

随后，流程图可分叉为两个并列分支。左侧分支为“处理 running 中的 unfinished prefill”，右侧分支为“处理 waiting 队列中的新 prefill”。两条分支的内部判定逻辑保持一致：首先检查当前 active unfinished prefill 数是否已经达到上界 $C_t$；若已达到，则不接纳新的 active prefill，并将 running unfinished prefill defer 回 waiting，或直接停止 waiting 请求的接纳。若尚未达到上界，则调用基线调度器的时间预算搜索器得到候选 chunk $u_i^\star$，并进一步计算最小有效推进阈值 $m_i = \min(r_i, L_{\min})$。

在这两个分支内部，应设置一个统一的判定菱形框，即“候选 chunk 是否满足最小有效推进条件”。若满足 $u_i^\star \ge m_i$，则箭头指向“接纳为 active prefill”模块，并更新当前 batch。若不满足，则箭头继续指向下一判定框“当前 batch 中是否已有 active prefill”。若当前 batch 中尚无 active prefill，则触发 warm-start，使用 $\min(h_i, m_i)$ 作为启动 chunk，并将该请求接纳进入 batch；若当前 batch 中已经存在 active prefill，则该请求不进入本轮 batch，对 running unfinished prefill 执行 defer，对 waiting prefill 则保持其继续等待。

最后，左右两个分支在图底部汇合到“输出本轮调度 batch”模块。该模块可标注输出内容包括：scheduled decode、scheduled active prefill、deferred prefill，以及用于实验统计的控制行为指标。整个流程图应突出 APC 与基线时间预算调度器之间的关系，即 APC 并不替代原搜索器，而是在其外层增加 unfinished prefill 活跃度控制逻辑。

建议的图注可写为：

> 图 X 展示了 Active Prefill Control (APC) 在单轮调度中的整体流程。APC 在继承基线调度器 decode 优先顺序和时间预算搜索结果的基础上，引入 dynamic active-prefill cap、minimum meaningful chunk 和 warm-start 三个附加决策环节，从而控制 unfinished prefill 的活跃数量并维持非零 prefill 活性。

### 5.2 机制图文字说明

若需要绘制机制图，建议采用从左到右的结构布局，将 APC 的核心机制拆分为“约束输入”“控制逻辑”“调度结果”三个层次。最左侧为约束输入层，可包含四个输入框：`decode occupancy`、`seq-slot budget`、`token budget`、`remaining prompt length`。其中，`decode occupancy` 与 `seq-slot budget` 共同决定 unfinished prefill 的结构可用空间，`token budget` 决定本轮还能支撑多少 prefill token，`remaining prompt length` 决定单条 unfinished prefill 的最小有效推进需求。

中间层为 APC 的三个核心机制模块。第一个模块为 **Dynamic Active-Prefill Cap**，用于根据当前 decode 占用和剩余 token 预算计算 active unfinished prefill 上界 $C_t$。第二个模块为 **Minimum Meaningful Chunk Constraint**，用于判定候选 chunk 是否满足 $u_i^\star \ge m_i$，从而过滤掉大量仅能带来碎片化推进的小 chunk。第三个模块为 **Warm-start for Prefill Reactivation**，用于在当前 batch 中 active prefill 数量为零时，以最小代价重新注入一条 prefill，防止系统坍缩为长期纯 decode 运行。

最右侧为调度结果层，可用两个对比结果框展示 APC 带来的结构差异。其一是“Without APC”，可标注为：unfinished prefill 无约束增长或长期失活，batch 更容易出现 prefill 碎片化推进或纯 decode 尾部。其二是“With APC”，可标注为：unfinished prefill 活跃度受控、单条 prefill chunk 更具有效性、系统持续维持非零 prefill 活性。若希望图示更直观，可以在最右侧进一步补充两个小型 batch 示意图：基线情形下表现为大量 decode token 加若干零散的小 prefill chunk；APC 情形下表现为 decode 仍占主导，但只保留少量 active prefill，且单条 prefill 的 chunk 更完整。

该机制图的重点不在于逐行复现算法，而在于从结构上说明 APC 如何在不替代 target-time 搜索器的前提下，通过三个额外控制环节改变 unfinished prefill 的调度形态。也就是说，机制图应强调 APC 的方法定位是“外层结构控制”，而不是“内层 chunk 打分替换”。

建议的图注可写为：

> 图 Y 展示了 Active Prefill Control (APC) 的核心机制。APC 从 decode 占用、剩余 seq slot、token 预算和请求剩余 prompt 长度中提取约束信息，并通过动态 unfinished-prefill 上界、最小有效 chunk 判定和 warm-start 机制，对基线时间预算调度器形成外层结构控制，从而抑制 prefill 碎片化推进并避免系统长期退化为纯 decode 运行。
