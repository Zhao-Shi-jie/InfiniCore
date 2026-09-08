# Cambricon MLU CSR SpMM 优化实现

> 用途：硕士中期检查 PPT 中“已完成工作：CSR SpMM 优化实现”页面的技术材料。
>
> 本文只描述 MLU CSR SpMM kernel 的计算分解、片上存储、向量化和两层流水线，不涉及工程接口、上层封装或稀疏格式接入。

当前实现对应：

- `bang/spmm_bang.mlu`：连续列路径选择、NRAM 列块大小计算和 kernel launch。
- `bang/spmm_bang_kernel.mlu`：`spmmCsrF32ContiguousKernel` 的具体实现。
- `../../devices/bang/common_bang.h`：NRAM 容量和对齐要求。

## 1. 一页 PPT 可直接采用的内容

### 1.1 页面标题

**Cambricon MLU 上 CSR SpMM 的分块向量化与两层流水优化**

### 1.2 问题与优化目标

CSR SpMM 计算为：

```text
C = alpha * A * B + beta * C

A: [M, K]，CSR 稀疏矩阵
B: [K, N]，稠密矩阵
C: [M, N]，稠密输出
```

CSR 每行非零元数量不规则，`col_indices` 指向的 `B` 行通常也不连续。若逐输出元素计算，会重复遍历 CSR 行并产生大量离散 GDRAM 访问，难以利用 MLU 的变长向量指令和多指令流并行能力。

本实现的目标是：

1. 沿输出矩阵的连续列方向分块，将标量 SpMM 转换为 NRAM 向量乘加。
2. 使用 `B` 双缓冲，重叠下一个非零元的 `B` 行块加载与当前非零元的向量计算。
3. 使用 accumulator 双缓冲，重叠上一输出 tile 的写回与下一输出 tile 的首段计算。
4. 在 kernel 内融合 `alpha`、`beta`、尾块处理和空计算快速路径，减少额外 kernel 和无效访存。

### 1.3 核心计算分解图

```text
CSR 的第 m 行                          B 的一个连续列块

A[m, k0] = a0  -------------------->  B[k0, n0:n1] --\
A[m, k1] = a1  -------------------->  B[k1, n0:n1] ----+--> C[m, n0:n1]
A[m, k2] = a2  -------------------->  B[k2, n0:n1] --/

C[m, n0:n1]
    = alpha * sum_i(A[m, ki] * B[ki, n0:n1])
    + beta * C[m, n0:n1]
```

一个 MLU task 一次负责一个 `(output_row, output_column_tile)`。该 task 独占对应的 `C` tile，因此不需要原子操作或跨 Core 归约。

### 1.4 四缓冲 NRAM 设计

```text
240 KiB NRAM

| acc_ping | acc_pong | B_ping | B_pong |
     ^           ^          ^        ^
     |           |          |        |
  输出tile累加双缓冲      B行片段加载双缓冲
```

- `B_ping/B_pong`：非零元级 Load/Compute 指令流水。
- `acc_ping/acc_pong`：输出 tile 级 Compute/Store 软件流水。
- 向量长度按 128 字节对齐，当前最大 CSR 列块为 15,328 个 `float`。

### 1.5 两层流水线的 PPT 核心图

```text
时间 -------------------------------------------------------------------->

IO 流      Store C[t-1] -> Load B[t,1] | Load B[t,2] | ... | Load old C[t]
                 ||             ||             ||              ||
Compute 流 Compute B[t,0] | Compute B[t,1] | Compute B[t,2] | alpha * acc[t]
                 ||             ||             ||
Scalar 流   循环控制/地址计算   循环控制/地址计算   GPR 状态维护

tile 级重叠：Store C[t-1]       || Compute tile[t] 的首个非零元
nnz  级重叠：Load B[t,i+1]      || Compute B[t,i]
收尾阶段重叠：Load old C[t]     || alpha * accumulator[t]
```

需要在 PPT 中明确：`Load B` 和 `Store C` 都在 IO 流中，二者不能同时执行。当前最大并发是“一个 IO 操作 + 一个 Compute 操作 + Scalar 控制工作”，不是 Load、Compute、Store 三个硬件阶段同时运行。

### 1.6 可放在 PPT 页底部的总结

- 完成了基于输出列 tile 的 CSR SpMM 向量化 kernel。
- 完成了 `B` 行块 ping-pong，实现 nnz 级 `GDRAM2NRAM` 与向量乘加重叠。
- 完成了 accumulator ping-pong，实现上一 `C` tile 写回与下一 tile 首段计算重叠。
- 完成了 128 字节对齐、尾块零填充、`alpha/beta` 融合和 `beta == 0` 免读优化。
- 保持每个输出 tile 单 task 独占，不引入原子操作和跨 Core 归约。
- 当前限制是 CSR 行负载不均衡、每个非零元存在同步开销，以及 Load/Store 共享 IO 流。

### 1.7 建议的讲解稿

> 我针对 MLU 上最通用的 CSR SpMM 实现了连续列优化路径。首先沿输出矩阵的 N 维切分 column tile，一个 task 负责一行 A 与 B 的一个连续列块，最终独占一个 C tile，因此不需要原子操作。片上 NRAM 划分为两组双缓冲：B ping-pong 在非零元粒度上，让下一个 B 行片段的 GDRAM 到 NRAM 搬运与当前行片段的向量乘加在 IO 流和 Compute 流上重叠；accumulator ping-pong 在输出 tile 粒度上，让上一 tile 的异步写回与下一 tile 的首段计算重叠。两个层面的流水交织后，Compute 流在 tile 边界和 tile 内部都能被不同的 IO 操作覆盖。同时实现了 128 字节向量对齐、尾块保护以及 alpha、beta 融合。需要说明的是，B 的加载和 C 的写回共用 IO 流，二者仍然串行，当前实现并不是 Load、Compute、Store 三者完全并行。

## 2. CSR SpMM 的计算映射

### 2.1 CSR 行计算

CSR 使用三个数组：

```text
crow_indices[M + 1]
col_indices[nnz]
values[nnz]
```

对输出行 `m`，基础计算为：

```text
ptr_begin = crow_indices[m]
ptr_end   = crow_indices[m + 1]

for ptr in [ptr_begin, ptr_end):
    k = col_indices[ptr]
    C[m, :] += values[ptr] * B[k, :]
```

CSR 的 `values` 和 `col_indices` 在每行范围内紧凑存储，但 `k` 通常是不规则的，因此连续两个非零元可能访问距离很远的 `B` 行。

### 2.2 连续列快路径的条件

主要优化 kernel 要求：

```text
B.col_stride == 1
C.col_stride == 1
N > 1
```

这里的“连续”指 `B/C` 最后一维连续，不要求 `col_indices` 连续、有序或相邻。

即使 `B` 的行跨度包含 padding，只要列 stride 为 1，下面的行片段仍然连续：

```text
B[k, col_begin:col_end]
```

因此可以通过一次变长 `GDRAM2NRAM` 搬运把完整向量块送入 NRAM。

### 2.3 输出列分块

设列块宽度为 `col_block`：

```text
col_tiles  = ceil(N / col_block)
total_tiles = M * col_tiles
```

线性 tile 编号映射为：

```text
row       = tile / col_tiles
col_begin = (tile % col_tiles) * col_block
curr_cols = min(col_block, N - col_begin)
```

一个 tile 的数学结果是：

```text
acc[0:curr_cols] = 0

for ptr in CSR row:
    k = col_indices[ptr]
    acc += values[ptr] * B[k, col_begin:col_begin + curr_cols]

C[row, col_begin:col_begin + curr_cols]
    = alpha * acc + beta * old_C_tile
```

每个 task 从 `taskId` 对应的 tile 开始，再以 `taskDim` 为步长继续处理：

```text
tile, tile + taskDim, tile + 2 * taskDim, ...
```

后文时序图为了简洁使用 `t`、`t+1` 表示“同一 task 先后处理的两个 tile”；源码中的实际编号关系是 `next_tile = tile + taskDim`，不一定是全局线性编号相邻的两个 tile。

该划分具有以下特点：

- 一个输出 tile 只由一个 task 写入，不需要原子加。
- 不需要把一条 CSR 行沿 nnz 切开，因此没有跨 task 的部分和归约。
- 不需要预先整理右矩阵 `B`。
- 当一行对应多个 column tile 时，每个 tile 都会重新遍历该 CSR 行的元数据。

## 3. NRAM 规划与向量化

### 3.1 四缓冲布局

项目为每个 MLU Core 提供 240 KiB NRAM，CSR 连续列 kernel 使用：

```text
nram_acc_ping : 当前或下一输出 tile 的累加器
nram_acc_pong : 另一输出 tile 的累加器
nram_b_ping   : 当前或下一非零元对应的 B 行片段
nram_b_pong   : 另一非零元对应的 B 行片段
```

四个区域大小都为 `col_block * sizeof(float)`：

```text
NRAM usage = 4 * col_block * sizeof(float) + alignment reserve
```

按照 240 KiB NRAM 和 128 字节对齐计算，当前：

```text
csr_col_block = 15,328 float
```

四缓冲是两层流水的必要存储条件，同时也将单个列 tile 的最大宽度从双缓冲方案的约 30,688 个 `float` 降低到 15,328 个 `float`。

### 3.2 向量指令

对一个非零元 `A[row,k] = value`，当前 Compute 流执行：

```text
nram_b_current *= value
nram_acc_current += nram_b_current
```

对应 BANG C 向量操作：

```cpp
__bang_mul_scalar(nram_b_current,
                  nram_b_current,
                  value,
                  aligned_cols);

__bang_add(nram_acc_current,
           nram_acc_current,
           nram_b_current,
           aligned_cols);
```

向量指令直接以 NRAM 地址作为源和目的地址。NRAM 已经是 MLU Core 的片上存储，不存在一个需要用户显式编写的“NRAM 再加载到 Core 寄存器”阶段。

### 3.3 128 字节对齐和尾块

最后一个 tile 可能小于 `col_block`，计算长度向 128 字节对齐：

```text
aligned_cols = align_up(curr_cols * sizeof(float), 128) / sizeof(float)
```

处理方式为：

1. accumulator 按 `aligned_cols` 清零。
2. 尾 tile 的两个 `B` 缓冲按 `aligned_cols` 清零。
3. GDRAM 到 NRAM 只搬运 `curr_cols` 个有效元素。
4. Compute 流按 `aligned_cols` 执行向量操作。
5. NRAM 到 GDRAM 只写回 `curr_cols` 个有效元素。

这样既满足向量指令对齐要求，也不会越界访问 `B/C`。

## 4. 第一层流水：nnz 级 Load/Compute 指令流水

### 4.1 目标

CSR 行中每个非零元都需要读取一个新的 `B` 行片段：

```text
B[k_i, col_begin:col_end]
```

若采用串行执行：

```text
Load B[i]
等待
Compute B[i]
Load B[i + 1]
等待
Compute B[i + 1]
```

则 IO 流搬运时 Compute 流空闲，Compute 流计算时 IO 流空闲。

因此使用 `B_ping/B_pong` 将当前计算数据和下一次加载目标分离。

### 4.2 Prologue：启动流水

在进入非零元循环前：

1. 从 `crow_indices` 取得当前 CSR 行范围。
2. 读取第一个稀疏列号 `k_0` 和数值 `a_0`。
3. 预读第二个列号 `k_1`，用于下一次异步搬运。
4. 将 `B[k_0, tile_cols]` 异步搬入 `B_ping`。
5. 使用 `__sync_io()` 等待第一个 B 行片段就绪。

第一份数据没有前序计算可以覆盖，因此首次加载必须单独完成，这是流水线启动开销。

### 4.3 Steady state：稳定流水

假设当前 `B[i]` 位于 `B_ping`，`B_pong` 空闲：

```text
IO 流：       Load B[i + 1] -> B_pong
Compute 流：  B_ping *= value[i]
              accumulator += B_ping
Scalar 流：   处理 GPR 中的 value、列号、地址和循环状态
```

下一轮交换 ping/pong：

```text
IO 流：       Load B[i + 2] -> B_ping
Compute 流：  B_pong *= value[i + 1]
              accumulator += B_pong
```

稳定时序为：

```text
时间 ------------------------------------------------------------------>

IO       Load B1          Load B2          Load B3          Load B4
            ||               ||               ||               ||
Compute Compute B0        Compute B1        Compute B2        Compute B3
            ||               ||               ||               ||
Scalar  prepare k1       prepare k2       prepare k3       prepare k4
```

这里的重叠来自 MLU 的硬件多指令流：

- `__memcpy_async(..., GDRAM2NRAM)` 发射到 IO 流。
- `__bang_mul_scalar` 和 `__bang_add` 发射到 Compute 流。
- 循环控制、地址计算和 GPR 状态处理主要位于 Scalar 流。

`values/col_indices/crow_indices` 位于 GDRAM，因此元数据的实际读取仍属于 IO 流；Scalar 流负责生成地址并消费读取到 GPR 的结果。硬件维护 GPR 依赖，但这些小粒度元数据读取仍会与 B/C 的大块搬运共享 IO 资源。

ping-pong 本身只负责解除 NRAM 地址冲突；真正让指令同时执行的是 IO 流和 Compute 流的硬件并行能力，以及将同步推迟到数据真正被消费的位置。

### 4.4 同步位置

内层流水使用两类同步：

```text
__sync_compute()
```

在 IO 流准备覆盖一个 B 缓冲前调用，保证 Compute 流已经不再读取这个地址。

```text
__sync_io()
```

在 Compute 流准备消费新 B 缓冲前调用，保证 GDRAM 到 NRAM 的异步搬运已经完成。

核心顺序可概括为：

```cpp
__sync_compute();
__memcpy_async(b_next, B_next, bytes, GDRAM2NRAM);

__bang_mul_scalar(b_current, b_current, value, length);
__bang_add(accumulator, accumulator, b_current, length);

// 到下一轮真正使用 b_next 前才执行 __sync_io()
```

异步 Load 发射后没有立即等待，因此 `Load B[i+1]` 和 `Compute B[i]` 之间形成重叠窗口。

### 4.5 Epilogue：排空流水

最后一个非零元没有后继 B 行片段可加载，因此只执行最后一次向量乘加，并通过 `__sync_compute()` 等待累加完成。

完整内层流水为：

```text
Prologue: Load B0
Steady:   Load B[i+1] || Compute B[i]
Epilogue: Compute Blast
```

## 5. 第二层流水：output tile 级 Compute/Store 软件流水

### 5.1 Tile 的三个逻辑阶段

每个 output tile 在逻辑上包含：

#### Load/Prepare

- 计算 `row`、`col_begin`、`curr_cols`。
- 获取 CSR 行起止位置。
- 清零 accumulator。
- 读取第一个稀疏值和列号。
- 预取第一个 `B` 行片段。

这里不会一次加载整个 tile 所需的全部 B 数据；剩余 B 行片段由内层流水逐项加载。

#### Compute

- 遍历完整 CSR 行。
- 在内层流水中执行 `Load B[i+1] || Compute B[i]`。
- 完成 `alpha * accumulator + beta * old_C`。

#### Store

- 将 accumulator 的 `curr_cols` 个有效元素异步写回对应的 `C` tile。

逻辑依赖是：

```text
Load/Prepare tile[t] -> Compute tile[t] -> Store tile[t]
```

但当前硬件上 Load B 和 Store C 都使用同一个 IO 流，因此三者不能全部同时执行。当前外层流水更准确地说是“带下一 tile 首块预取的 Compute/Store 双缓冲流水”。

### 5.2 为什么需要 accumulator ping-pong

如果只有一个 accumulator：

```text
Store C[t] 正在从 accumulator 读取
```

下一 tile 就不能清零或修改同一块 NRAM，只能等待 Store 完成。

双 accumulator 将地址分离：

```text
acc_ping：上一 tile 已完成，IO 流正在将其写回 C
acc_pong：当前 tile 正在由 Compute 流累加
```

因此可以实现：

```text
Store C[t - 1] || Compute tile[t]
```

### 5.3 下一 tile 的预热

完成当前 tile `t-1` 后，kernel 在发射 Store 前先准备 tile `t`：

1. 选择另一个 accumulator。
2. 确认它上一次对应的 Store 已完成。
3. 清零新 accumulator。
4. 计算 tile `t` 的行号和列范围。
5. 读取第一个稀疏值、首个列号和第二个列号。
6. 将 `B[t,0]` 预取到 `B_ping` 并等待就绪。
7. 异步发射 `Store C[t-1]`。
8. 立即使用已就绪的 `B[t,0]` 开始 tile `t` 的首段 Compute。

预取必须放在 Store 前面。若顺序为：

```text
Store C[t-1] -> Load B[t,0] -> Compute B[t,0]
```

由于 Store 和 Load 同属 IO 流，Compute 必须等待两个 IO 操作。

当前顺序为：

```text
Preload B[t,0] -> Store C[t-1]
                          ||
                     Compute B[t,0]
```

第一份 B 数据已经位于 NRAM，因此 `Store C[t-1]` 运行时 Compute 流可以立刻处理 tile `t` 的第一个非零元。

### 5.4 外层流水的稳态

```text
时间 ------------------------------------------------------------------>

IO       Store C[t-1]                         Store C[t]
              ||                                  ||
Compute  Compute first contribution of t     Compute first contribution of t+1

acc      acc_ping: store                     acc_pong: store
         acc_pong: compute                   acc_ping: compute
```

`output_store_pending` 用于记录是否仍存在未完成的异步 Store。当 accumulator 再次轮转回来准备清零时，如果 Store 尚未完成，必须先执行 `__sync_io()`，避免 Compute 流覆盖 IO 流仍在读取的数据。

## 6. 两层流水交织后的完整稳态

### 6.1 稳态前提

为了同时观察两层流水，假设：

- 当前 task 后面仍有 output tile。
- 每个 CSR 行至少包含多个非零元。
- `N > 1` 且 `B/C` 最后一维连续。
- 当前 tile 的第一个 B 行片段已经预取完成。

假设 tile `t-1` 使用 `acc_ping`，tile `t` 使用 `acc_pong`。

### 6.2 整体时序

```text
时间 ---------------------------------------------------------------------------------->

IO 流
  Store C[t-1]
       -> Load B[t,1]
       -> Load B[t,2]
       -> ...
       -> Load old C[t]
       -> Preload B[t+1,0]
       -> Store C[t]
       -> Load B[t+1,1]
       -> ...

Compute 流
  Compute B[t,0]
       -> Compute B[t,1]
       -> Compute B[t,2]
       -> ...
       -> alpha * acc[t]
       -> beta * old_C[t] + acc[t]
       -> Compute B[t+1,0]
       -> Compute B[t+1,1]
       -> ...

Scalar 流
  tile/row 地址计算、CSR 循环、GPR 状态处理、ping-pong 状态切换
```

其中 `value/column/crow` 的 GDRAM 读取归入 IO 流；上图的 Scalar 流只表示地址、循环和读取结果的标量处理。

把发生重叠的区间横向对齐，可以得到：

| 执行区间 | IO 流 | Compute 流 | 使用的双缓冲 |
| --- | --- | --- | --- |
| tile 边界 | `Store C[t-1]`，随后同一 IO 队列执行 `Load B[t,1]` | `Compute B[t,0]` | `acc_ping/pong`，同时开始使用 `B_ping/pong` |
| tile 内部 | `Load B[t,i+1]` | `Compute B[t,i]` | `B_ping/pong` |
| tile 收尾 | `Load old C[t]` | `alpha * acc[t]` | 复用空闲的 `B_ping` 保存旧 C |
| 下一边界 | `Store C[t]`，随后执行 `Load B[t+1,1]` | `Compute B[t+1,0]` | accumulator 角色交换 |

### 6.3 两层流水不是两套独立硬件

两层流水的关系是：

- output tile 流水决定跨 tile 的工作组织和 accumulator 生命周期。
- nnz 指令流水决定单个 tile 内相邻非零元的 B 缓冲生命周期。
- 两层流水都依赖同一组 IO、Compute 和 Scalar 硬件指令流。
- 两组 ping-pong 分别解除 tile 间和 nnz 间的 NRAM 地址冲突。

因此“两个层面的流水同时稳定”并不表示存在两条 IO 流或两套 Compute 流，而是 Compute 流在不同阶段由不同 IO 操作进行延迟隐藏：

```text
tile 边界：Store previous C || Compute current first B
tile 内部：Load next B      || Compute current B
tile 收尾：Load old C       || Scale current accumulator
```

### 6.4 明确不能重叠的过程

当前实现中以下操作不能并行：

- `Load B` 与 `Store C`：都属于 IO 流，按发射顺序串行。
- 两个不同非零元的向量乘加：都属于 Compute 流，串行执行。
- `beta * old_C` 与将其加到 accumulator：存在数据依赖，且都属于 Compute 流。
- accumulator 清零与使用同一 accumulator 计算：同一地址存在依赖。
- IO 覆盖某个 B ping/pong 与 Compute 读取同一缓冲：必须通过同步和双缓冲避免。

当前没有使用 Move 流、Cluster SRAM 或 Memory Core，因此不能表述为 IO、Move、Compute 的三级硬件并行。

## 7. Tile 收尾阶段的额外指令重叠

完成 CSR 累加后，需要计算：

```text
C_tile = alpha * accumulator + beta * old_C_tile
```

当 `beta != 0` 时：

```text
IO 流：       Load old C_tile -> 空闲的 B_ping
Compute 流：  accumulator *= alpha（仅 alpha != 1）
```

等待 old C 到达后：

```text
B_ping *= beta
accumulator += B_ping
```

当 `beta == 0` 时，不读取旧 C，直接跳过整段 GDRAM Load。当 `alpha == 1` 时，跳过 accumulator 缩放。

这部分优化减少了：

- 独立 scale/add kernel 的 launch 开销。
- 中间结果写回和重新读取。
- `beta == 0` 情况下完全无用的 C 读取。

## 8. 当前已实现的其他 CSR 优化

### 8.1 N 维向量化

每个非零元不再只贡献一个标量，而是一次处理整个连续列 tile，使变长 BANG 向量指令具备足够工作量。

### 8.2 输出 tile 独占

任务沿 `(row, col_tile)` 划分，不拆分同一个输出 tile 的 nnz 范围，因此避免：

- 原子加。
- 跨 Core 部分和归约。
- 额外 workspace。
- 归约次序变化引起的额外数值误差。

### 8.3 首值和第二列索引 lookahead

下一 tile 在 Store 前预先读取：

- 第一个非零值。
- 第一个 B 行号。
- 第二个 B 行号。

这样下一 tile 的第一次 Compute 和第二次 B Load 都可以尽早发射，减少小粒度元数据访问导致的流水气泡。

### 8.4 空计算快速路径

当 `alpha == 0` 或 `nnz == 0` 时，不遍历 CSR，也不读取 B，直接计算：

```text
C = beta * C
```

### 8.5 `N == 1` 专用路径

当 `N == 1` 时，SpMM 退化为 SpMV。此时列 tile 长度为 1，四缓冲、向量指令和同步开销通常得不偿失，因此使用按行标量累加的专用 kernel。

### 8.6 通用 stride fallback

当 `B/C` 的列 stride 不为 1 时，无法直接连续搬运列 tile，使用逐输出元素计算的通用路径保证正确性。该路径不是当前性能优化目标。

## 9. 优化收益的来源

在不填入未经测量的加速比时，可以从以下结构性变化说明收益来源：

| 优化 | 原始问题 | 当前实现的作用 |
| --- | --- | --- |
| 输出列分块 | 标量输出无法利用宽向量 | 将一个非零元扩展为整段向量乘加 |
| 连续 B 行片段 DMA | 大量逐元素 GDRAM 读取 | 使用变长连续搬运提高有效带宽 |
| B ping-pong | Load B 和 Compute 串行 | 重叠 `Load B[i+1]` 与 `Compute B[i]` |
| accumulator ping-pong | Store C 阻塞下一 tile | 重叠 `Store C[t-1]` 与 `Compute tile[t]` |
| alpha/beta 融合 | 需要额外 kernel 或中间写回 | 在 accumulator 写回前完成融合 |
| beta==0 免读 | 无效读取旧 C | 降低 GDRAM 流量 |
| 对齐和尾块保护 | 向量长度不规则、可能越界 | 统一向量执行长度，仅写回有效元素 |
| tile 独占 | 拆分 nnz 需要原子或归约 | 保持单写者，简化同步和数值路径 |

实际 PPT 若要展示性能结果，应补充同一硬件、同一矩阵集合上的 profiler 数据，例如：

```text
- baseline 与优化 kernel latency
- 不同 N 下的加速比
- 不同行平均 nnz 和方差下的性能
- IO/Compute 利用率
- GDRAM 带宽和同步等待占比
```

在没有这些数据前，不应把理论重叠直接表述为固定加速比。

## 10. 当前实现的限制

### 10.1 Load 和 Store 共享 IO 流

`GDRAM2NRAM` 和 `NRAM2GDRAM` 都属于 IO 流，因此：

```text
Store C -> Load B
```

只能顺序执行。外层流水只能利用 Store 与 Compute 的重叠，不能形成 Load、Compute、Store 三个完全独立的硬件阶段。

### 10.2 每个非零元存在同步开销

内层流水需要在缓冲复用和消费位置执行 `__sync_compute()`、`__sync_io()`。当 `N` 很小、列 tile 很短或行 nnz 很少时，同步开销可能大于被隐藏的 IO 延迟。

### 10.3 CSR 行负载不均衡

task 按输出 tile 数量分配，但每个 tile 的计算量与对应 CSR 行的 nnz 成正比。少数超长行会造成 Core 间负载不均衡和尾部延迟。

### 10.4 四缓冲降低单 tile 宽度

四个等长 NRAM 缓冲将最大列 tile 降至 15,328 个 `float`。当 N 很大时，同一输出行会被拆成更多 tile，并重复读取 CSR 元数据。

### 10.5 CSR 元数据仍为小粒度访问

`values` 和 `col_indices` 当前仍由循环逐项从 GDRAM 访问，没有成块搬入 NRAM。这些访问会与 B/C 搬运共享 IO 资源，并增加 Scalar 与 IO 之间的依赖。

### 10.6 B 行之间仍是随机访问

每个 `B[k, tile_cols]` 内部是连续的，但不同 `k` 由 CSR 列索引决定，行与行之间通常不连续。当前没有跨输出行的 B 缓存或复用机制。

## 11. 后续优化方向

适合在 PPT 的“下一步工作”中概括为：

1. 根据 `N`、平均行 nnz 和 tile 数量，在流水 kernel 与简单向量 kernel 之间自适应选择，避免短向量同步退化。
2. 将一段 `values/col_indices` 成块搬入 NRAM，减少小粒度元数据访问。
3. 采用 nnz-aware 任务划分或长行拆分，改善 CSR 行长度不均衡。
4. 研究更深的 B buffer ring 或多 nnz 一组的软件 stage，降低每个非零元的同步频率。
5. 根据目标 MLU 和 BANG C SDK 能力评估向量 FMA/AXPY，减少当前乘法和加法两条 Compute 指令及 NRAM 流量。
6. 研究 Memory Core、Cluster SRAM 和 MLU Core 协作的数据搬运路径，但需要额外 SRAM 双缓冲和 Cluster 级同步。

## 12. 答辩表述注意事项

建议使用以下准确表述：

- “使用 B 双缓冲，在非零元粒度重叠下一 B 行片段加载和当前向量乘加。”
- “使用 accumulator 双缓冲，在 output tile 粒度重叠上一 tile 写回和下一 tile 首段计算。”
- “逻辑上 tile 包含 Load、Compute、Store 三个阶段，硬件上当前主要实现 IO 与 Compute 的两流并行。”
- “NRAM 已经是 MLU Core 片上存储，向量指令直接读取 NRAM 地址。”
- “Load B 和 Store C 共用 IO 流，不能彼此同时执行。”

不建议使用以下表述：

- “Load B、Compute、Store C 三者同时执行。”
- “B 从 NRAM 再显式加载到 Core 后计算。”
- “当前已经使用 Move 流、Memory Core 或 Cluster SRAM。”
- “ping-pong 本身就能自动产生并行。”

更准确的关系是：

```text
ping-pong：解除地址冲突
async copy：允许延迟等待
IO/Compute 多指令流：提供真实硬件并行
sync：在数据消费或缓冲复用位置维护正确性
```

## 13. PPT 页面排版建议

建议一页分为三个区域：

### 左侧：计算分块

```text
A 的一行 + B 的连续列块 -> C 的一个 tile
```

突出：

- `(row, col_tile)` 任务划分。
- 无原子、无跨 Core 归约。
- N 维向量化。

### 中间：NRAM 四缓冲

```text
| acc ping | acc pong | B ping | B pong |
```

分别用两种颜色表示：

- accumulator 双缓冲：tile 级流水。
- B 双缓冲：nnz 级流水。

### 右侧：组合时序

```text
Store C[t-1]    || Compute B[t,0]
Load B[t,i+1]   || Compute B[t,i]
Load old C[t]   || alpha * acc[t]
```

页底放一句限制：

```text
Load B 与 Store C 共用 IO 流，当前最大并发为 IO + Compute + Scalar。
```
