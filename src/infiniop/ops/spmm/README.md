# Cambricon MLU SpMM Kernel 实现与优化路线

本文档只记录 Cambricon MLU SpMM kernel 的计算分解、任务划分、片上内存规划、指令流调度、当前优化和后续优化方向。重点是 CSR kernel。

文档中的“当前实现”以以下文件为准：

- `bang/spmm_bang.mlu`：运行时路径选择、kernel launch 和 NRAM tile 规划。
- `bang/spmm_bang_kernel.mlu`：MLU BANG C kernel。
- `../../devices/bang/common_bang.h`：NRAM 容量、对齐和 MLU 公共定义。

## 1. 计算定义

SpMM 计算如下：

```text
C = alpha * A * B + beta * C
```

其中：

- `A` 是形状为 `[M, K]` 的稀疏矩阵。
- `B` 是形状为 `[K, N]` 的稠密矩阵。
- `C` 是形状为 `[M, N]` 的稠密矩阵。
- `alpha` 和 `beta` 是 host 侧传入的 `float` 标量。

## 2. MLU 硬件特征与当前映射

### 2.1 存储层次

当前实现涉及以下存储资源：

- GDRAM：保存稀疏矩阵元数据、`values`、`B` 和 `C`。
- NRAM：每个 MLU Core 私有，用于保存 `B` 的列块、`C` 的累加块和临时向量。
- GPR：保存行号、列号、offset、当前稀疏值等标量状态。

当前实现没有使用：

- Cluster SRAM。
- WRAM。
- Memory Core 主动搬运路径。
- MLU Core 的 SRAM/NRAM Move 流。

因此当前核心数据流是：

```text
GDRAM -> NRAM -> vector compute -> GDRAM
```

而不是更深的：

```text
Memory Core: GDRAM -> SRAM
MLU Core:                SRAM -> NRAM -> compute -> GDRAM
```

### 2.2 指令流

MLU Core 提供 IO、Move、Compute 和 Scalar 指令流。当前实现主要使用：

- IO 流：`B`/`C` 在 GDRAM 和 NRAM 之间搬运。
- Compute 流：`__bang_write_value`、`__bang_mul_scalar` 和 `__bang_add`。
- Scalar 流：CSR/ELL/SELL 循环、地址计算、分支和稀疏元数据处理。

CSR 连续列快路径通过 `__memcpy_async`、`__sync_io()` 和 `__sync_compute()` 显式重叠 IO 流与 Compute 流。其他格式的连续列快路径仍以同步搬运为主。

需要注意，“指令流水线”和“数据搬运/核上计算流水线”不是两项独立优化：

- MLU 的多条指令流是硬件执行资源。
- Load/Compute 流水是软件对工作顺序的安排。
- ping-pong 是解决 NRAM buffer 读写冲突的存储策略。

### 2.3 Task 和 launch

host 侧按照 `core_per_cluster` 和 `cluster_count` 构造 launch 维度，并使用 `cnrtFuncTypeUnion1`。输出工作以 `taskId` 为单位分发。

虽然使用了 Union1 launch，当前 kernel 中没有 `__is_mpu()` 分支、Cluster SRAM 或 Cluster 同步。因此不能把当前实现视为已经使用 Memory Core 或 Cluster 协作；它目前主要依赖每个 MLU Core 的私有 NRAM 和指令流。

后续应比较 Block 与 Union1 的实际性能。若 kernel 不使用 SRAM 和 Memory Core，Union1 不一定始终优于 Block。

### 2.4 NRAM 和对齐

项目当前定义：

```text
NRAM_MAX_SIZE = 240 KiB
ALIGN_SIZE = 128 bytes
```

连续列 kernel 将向量长度向 128 字节对齐，但写回 GDRAM 时只写 `curr_cols` 个有效元素，避免尾块越界。

ELL/SELL 连续列 kernel 使用：

```text
NRAM = accumulator + dense_input
```

对应的最大列块约为 30,688 个 `float`。

CSR 组合流水线使用：

```text
NRAM = accumulator_ping + accumulator_pong
     + dense_input_ping + dense_input_pong
```

对应的最大列块为 15,328 个 `float`。双 accumulator 支持上一 tile Store 与下一 tile Compute 重叠，双 dense input 支持相邻非零元之间的 B-load/Compute 重叠。四缓冲提高了流水覆盖范围，但也进一步缩小了单个 tile，这是当前 CSR 组合流水线最直接的资源代价。

## 3. 公共 Kernel 路径

### 3.1 最后一维连续快路径

当：

```text
B.col_stride == 1 && C.col_stride == 1
```

实现沿 `N` 维进行 NRAM 分块。每个 task 处理一个 `(row, col_tile)`：

1. 将 NRAM 累加区清零。
2. 遍历该稀疏行中的非零元或 packed slot。
3. 将对应的 `B[k, col_begin:col_end]` 连续搬入 NRAM。
4. 用稀疏标量值乘以该向量并累加。
5. 融合 `alpha` 和 `beta`。
6. 将有效列写回 `C`。

该路径利用了 MLU 的变长向量指令，避免让一个 task 只计算一个标量元素。

### 3.2 通用 stride 路径

当 `B` 或 `C` 最后一维不连续时，当前实现按 `[M, N]` 展平后分发，每个 task 以 scalar loop 计算一个或多个输出元素。

该路径保证功能完整，但没有：

- 将 strided GDRAM 数据 gather 到连续 NRAM。
- BANG 向量化。
- IO/Compute 流水。

因此它是 correctness fallback，不是性能目标路径。

### 3.3 `N == 1` 专用路径

当 `N == 1` 且最后一维连续时，所有格式都切换到 SpMV 风格 kernel：每个 task 负责若干行，直接使用标量累加。

这避免了：

- NRAM 大块规划。
- 对单个元素调用向量搬运和向量指令。
- 二维 tile 地址计算。

该路径仍会融合 `alpha` 和 `beta`。

### 3.4 空计算快速返回路径

当 `alpha == 0` 或逻辑 `nnz == 0` 时，不读取稀疏矩阵和 `B`，直接计算：

```text
C = beta * C
```

连续列布局使用 NRAM 向量 kernel，通用 stride 布局使用 scalar kernel。该优化避免了完全无效的稀疏遍历和随机 `B` 访问。

### 3.5 `alpha`/`beta` 融合

正常 SpMM kernel 在累加完成后直接执行：

```text
C_tile = alpha * accumulator + beta * C_tile
```

当 `beta == 0` 时不读取旧 `C`。当 `alpha == 1` 时跳过额外的 accumulator 缩放。

当前这些判断仍是运行时分支。常见组合可以进一步使用模板特化或独立 kernel，减少分支和不必要的指令发射。

## 4. CSR 当前实现

CSR 是当前最通用、最重要的格式，也是后续优化重点。

### 4.1 存储和访问

CSR 使用：

- `crow_indices[M + 1]`：每行在 packed arrays 中的起止位置。
- `col_indices[nnz]`：每个非零元对应的 `K` 维列号。
- `values[nnz]`：非零值。

对输出行 `m`：

```text
for ptr in crow_indices[m] .. crow_indices[m + 1]:
    k = col_indices[ptr]
    C[m, :] += values[ptr] * B[k, :]
```

CSR 没有 ELL/SELL 的显式填充开销，并能表示任意行长分布，因此存储最通用。但每行工作量不规则，`B[k, :]` 的行号也通常不连续。

### 4.2 三类 CSR kernel

#### 连续列 kernel

`spmmCsrF32ContiguousKernel` 是主要优化路径，适用于 `N > 1` 且 `B/C` 最后一维连续。

并行单元是 `(output_row, output_column_tile)`。不同 task 独占不同的 `C` tile，因此：

- 不需要原子操作。
- 不需要跨 core 归约。
- `beta * C` 可以直接在拥有该 tile 的 task 中融合。

#### 通用 stride kernel

`spmmCsrF32StridedKernel` 按输出元素并行，逐元素遍历整条 CSR 行。它支持任意非零 stride，但同一行的 CSR 元数据和 `values` 会被不同输出列重复读取，且不使用向量指令。

#### `N == 1` kernel

`spmmCsrF32SpmvKernel` 按行并行。每个输出行只由一个 task 计算，无原子操作。

### 4.3 CSR 的两层 NRAM 流水线

连续列 kernel 将指令级流水和传统 Load/Compute/Store 流水交织在一起。NRAM 划分为：

```text
nram_acc_ping : 当前或下一 C 列块的累加器
nram_acc_pong : 当前或下一 C 列块的累加器
nram_b_ping   : 当前或下一非零元对应的 B 行块
nram_b_pong   : 当前或下一非零元对应的 B 行块
```

两组 ping-pong 的职责不同：

- `nram_b_ping/pong` 构成非零元内层的指令流水线。
- `nram_acc_ping/pong` 构成输出 tile 外层的传统三阶段流水线。

与沿 CSR 非零元流切 chunk 的 SpMV 三阶段实现不同，本实现沿输出 `N` 维切 column tile。每个 output tile 都从 `crow_indices[row]` 到 `crow_indices[row + 1]` 遍历完整 CSR 行，并直接加载 `B[k, col_begin:col_end]`，因此：

- 不需要预先整理右矩阵数据。
- 不会把一条 CSR 行截断在两个 nnz chunk 之间。
- 不需要为不完整行额外 load 或跨 chunk 求和。
- 代价是同一输出行有多个 column tile 时会重复遍历 CSR metadata。

#### Prologue

1. 清零第一个 accumulator。
2. 初始化尾块 padding。
3. 等待 Compute 流完成初始化。
4. 预读第一个稀疏值、第二个列索引，并异步预取第一个 `B` 行块到 ping。
5. 等待第一个 `B` 行块就绪。

#### 内层指令流水线

对第 `i` 个非零元：

```text
IO stream:       load B[i + 1] into the free buffer
Compute stream:  accumulator += value[i] * B[i]
Scalar stream:   load/compute value[i], col[i + 1], pointers and loop state
```

`__sync_io()` 保证当前 ping/pong 已经搬运完成；`__sync_compute()` 保证将要被下一次 IO 复用的 buffer 不再被 Compute 流读取。

当前稀疏值 `values[i]` 在发射下一次大块 `B` 搬运之前读取，减少 Compute 流因标量值尚未就绪而等待下一次 IO 的概率。

#### 外层传统三阶段流水线

完成当前输出 tile 的稀疏累加后：

1. 融合当前 tile 的 `alpha` 和 `beta`。
2. 选择另一个 accumulator，等待它上一次 Store 完成后清零。
3. 预读下一 tile 的首个稀疏值和第二个列索引。
4. 将下一 tile 的第一个 `B` 行块预取到 `nram_b_ping` 并确认就绪。
5. 对当前 accumulator 发射异步 NRAM 到 GDRAM Store，但不立即同步。
6. 进入下一轮后，使用已经就绪的 `B`、稀疏值和列索引直接发射首段 Compute。

稳态关系如下：

```text
IO stream:       Store C[t - 1] ---- Load B[t, 1] ---- Load B[t, 2] ...
Compute stream:  Compute first contribution of tile t ---- remaining contributions
Scalar stream:   loop/address work for tile t and lookahead metadata for tile t + 1
```

`B[t, 0]` 以及首个稀疏值在 `Store C[t - 1]` 发射前已经就绪。因此下一 tile 的第一段 Compute 不依赖正在执行的 Store，可以与它重叠。第二个 `B` load 会排在 Store 后面的同一 IO 流中，并继续参与内层 B ping-pong。

当某个 task 没有下一 tile 时，kernel 在退出前等待最后一次 Store。若下一次准备复用的 accumulator 仍有 Store 未完成，也会在清零前显式等待。

#### 当前 tile 的收尾

1. 等待最后一次向量累加完成。
2. `beta != 0` 时异步读取旧 `C`。
3. Compute 流同时执行 `alpha * accumulator`。
4. 等待旧 `C` 就绪，执行 `beta * C` 并累加。
5. 等待 Compute 流完成。
6. 下一 tile 存在时提前准备下一 tile；随后异步写回当前 `C`，将同步推迟到依赖该 Store 的位置。

### 4.4 CSR 流水线实际利用到的硬件能力

当前流水线利用了：

- IO 流与 Compute 流可并行执行。
- BANG 变长向量指令可以对完整 `N` tile 执行乘法和加法。
- NRAM 是每个 core 私有的低延迟向量存储。
- Scalar 流与其他流之间的 GPR 依赖由硬件维护。
- 双 accumulator 解除上一 tile Store 与下一 tile Compute 对同一 NRAM 地址的冲突。
- 首值和第二列索引 lookahead，避免下一 tile 首段 Compute 因小粒度 metadata load 排在 Store 后面而失去重叠。

当前流水线没有利用：

- Move 流。
- Cluster SRAM。
- Memory Core。
- MLU Core 间数据共享。

它也不能让 GDRAM Load 与 Store 彼此并行，因为二者都属于 IO 流，仍按发射顺序串行。

所以当前实现同时包含“Core 内 B-load/Compute 指令流水”和“输出 tile 级 Load/Compute/Store 软件流水”。传统三阶段的 Load 与 Store 不能彼此并行，但二者都可以在无 NRAM 依赖时与 Compute 流重叠。当前仍不是 Memory Core/MLU Core 四级或五级流水。

### 4.5 CSR 当前优势

- CSR 存储最通用，不要求行长规则或预排序。
- 无 padding，packed storage 大小等于逻辑 `nnz`。
- 连续列路径以向量块处理 `N`，适合中等和较大的 `N`。
- 每个输出 tile 独占，避免原子操作和归约误差。
- `B` load 与向量乘加有明确重叠机会。
- 上一 tile 的异步 Store 可以与下一 tile 的首段 Compute 重叠。
- 支持空行、`int32/int64` 索引、尾 tile、`beta == 0` 和任意非零行 stride。

### 4.6 CSR 当前限制

#### 行间负载不均衡

task 按输出 tile 数量均分，但每个 tile 的成本与对应 CSR 行的 nnz 成正比。少数超长行会形成明显的尾部延迟，尤其在 `N` 较小时，一条长行可并行的列 tile 数量不足。

#### 每个非零元都有同步开销

当前 steady state 对每个非零元至少执行一次 `__sync_io()`，并在 ping/pong 复用前执行 `__sync_compute()`。当 `N` 很小或每次向量计算很短时，同步成本可能高于被隐藏的 IO 延迟。

#### 四缓冲缩小列 tile

CSR tile 从两缓冲情况下约 30,688 个 `float` 降为 15,328 个 `float`。非常宽的 `N` 会增加 tile 数量、重复 CSR 元数据遍历，并增加 `B/C` DMA 事务数量。不同 column tile 覆盖互不重叠的 `N` 区间，因此这不会直接重复读取同一个 `B` 元素，但会使传输更碎片化。

#### 稀疏元数据未分块搬入 NRAM

`crow_indices`、`col_indices` 和 `values` 仍由 scalar loop 逐项访问 GDRAM。这会带来：

- 小粒度 GDRAM 事务。
- Scalar/IO 依赖。
- `int64` 索引的额外带宽。
- 与大块 `B` load 共享 IO 资源。

#### `B` 行访问不规则

虽然每次读取的 `B[k, col_begin:col_end]` 是连续块，但不同 `k` 由 CSR `col_indices` 决定，行与行之间通常是随机访问。当前没有：

- 对重复 `k` 的缓存。
- 对排序列索引的预取策略。
- 跨相邻输出行的 `B` 复用。

#### Compute 流需要两条向量指令

每个非零元当前执行：

```text
nram_b *= value
nram_acc += nram_b
```

这会读写一次临时 `nram_b`，再读取它进行加法。如果目标 MLU 架构和 BANG C SDK 提供适合的 fused AXPY/FMA primitive，可以减少 Compute 指令和 NRAM 流量。该能力需要在实际 SDK 上确认，不能仅按名称假设存在。

#### 外层三阶段流水线的覆盖范围有限

当前只提前准备下一 tile 的第一个 `B` 行块、首个稀疏值和第二个列索引。这样可以让上一 tile Store 与下一 tile 首段 Compute 重叠，但后续 `B` load 与 Store 位于同一 IO 流，不能同时执行。若一个 task 只分到一个 tile、下一 tile 是空行，或者首段向量很短，外层流水线收益会较小。

#### 通用 stride 路径性能弱

通用 stride kernel 对每个输出元素重复遍历同一 CSR 行，既没有元数据复用，也没有向量化。

#### host 端同步

launch wrapper 在 kernel 发射后调用 `cnrtQueueSync(queue)`，因此 host 需要等待本次计算完成。性能分析应分别记录纯 kernel latency 和包含 queue sync 的 launch latency；若要实现跨 kernel 异步重叠，需要统一检查错误传播和 stream 顺序，不能只删除这一处同步。

## 5. ELL 当前实现

### 5.1 存储布局

ELL 为每行分配固定 `ell_width`：

```text
ptr = row * ell_width + slot
```

packed storage 大小为：

```text
M * ell_width
```

短行使用零值填充。当前 kernel 通过 `value == 0` 跳过填充和真实零值，不读取对应的 `B` 行。

### 5.2 已实现优化

- 每行地址计算规则，不需要读取 `crow_indices`。
- 连续列时复用 NRAM accumulator 和 `B` tile 向量化。
- 零填充项在读取 `B` 前跳过，避免无效的大块 GDRAM load。
- `N == 1`、通用 stride、空计算和 `alpha/beta` 融合均有对应路径。

### 5.3 与硬件的关系

ELL 的固定行宽减少 Scalar 流中的不规则边界读取，适合行长接近的矩阵。当前 row-major packed layout 对单个 task 遍历一行很直接，但尚未让多个 core 协作处理同一 slot，也没有对 `B` load 使用异步 ping-pong。

### 5.4 当前限制

- `ell_width` 由全局最长行决定，长尾矩阵会产生大量 padding。
- 每个 padding slot 仍需要读取 `value`、判断和循环控制。
- `value == 0` 同时表示 padding 和合法零值，虽然数值结果正确，但格式语义不够显式。
- 连续列路径仍是 Load 后 Compute 的串行内循环。
- `ell_width` 没有根据矩阵分布或硬件自动选择。

## 6. SELL 当前实现

### 6.1 存储布局

SELL 将存储行按 `slice_height` 分组。每个 slice 使用自己的最大行宽，因此只在 slice 内进行 padding。

slice 内采用 slot-major 布局：

```text
ptr = slice_begin + slot * slice_height + row_in_slice
```

`slice_offsets` 给出每个 slice 的 packed storage 起止位置。

### 6.2 已实现优化

- 相比 ELL，将 padding 范围从全矩阵缩小到每个 slice。
- 同一 slot 下相邻存储行的 `values/col_indices` 连续，有利于相邻 core 同时处理相邻存储行时形成更规则的访问。
- `slice_width` 由 offset 差计算，不需要额外 row length 数组。
- 连续列时使用 NRAM 向量化。
- SELL 和 SELL-sigma-c 使用同一组模板 kernel，避免维护两套计算主体。

### 6.3 与硬件的关系

`slice_height` 可以与一个 Cluster 内的 MLU Core 数量建立关系，使同一 slice 的多行由邻近 core 处理。但当前 task 分配只是按 storage row 和 column tile 展平，没有显式保证一个 slice 由一个 Cluster 协作处理，也没有使用 SRAM 或 Cluster 同步。

### 6.4 当前限制

- `slice_height` 由用户或上层固定提供，没有硬件感知的自动调优。
- 单个 core 遍历自己的 storage row 时，packed metadata 地址步长为 `slice_height`。
- 当前没有以整个 slice 为单位协同搬运 metadata 或 `B`。
- padding slot 仍通过 `value == 0` 分支跳过。
- 连续列路径没有异步双缓冲。
- 行长在单个 slice 内仍可能高度不均衡。

## 7. SELL-sigma-c 当前实现

### 7.1 格式含义

SELL-sigma-c 在 SELL 基础上增加局部行重排：

- `c` 对应 `slice_height`。
- `sigma` 定义允许排序的行窗口。
- 每个 sigma 窗口内按行 nnz 排序，使长度相近的行进入同一 slice。
- `row_indices[storage_row]` 保存存储行到原始输出行的映射。

kernel 假定 storage rows 已在每个 sigma 窗口内按行长度排序，并通过 `row_indices` 提供原始输出行号；device kernel 本身不执行排序。

### 7.2 已实现优化

- 通过局部排序进一步减少 slice 内 padding。
- 相近行长进入同一 slice，改善按 storage row 分发时的负载均衡。
- 使用 `HAS_ROW_PERMUTATION=true` 模板特化；是否读取 `row_indices` 是编译期选择，不需要在内层循环运行时判断格式。

### 7.3 与硬件的关系

更均匀的 slice 宽度有利于相邻 MLU Core 在相似时间完成工作，降低 Cluster 内尾部空闲。代价是写 `C` 时按 `row_indices` 回到原始行，可能降低相邻 core 输出地址的连续性。

### 7.4 当前限制

- 行排序发生在 kernel 外，纯 kernel 性能不包含该预处理。
- `sigma` 只影响预处理；当前 device kernel 不直接使用该值。
- `sigma` 和 `slice_height` 没有自动选择。
- 行重排改善 padding，但可能恶化 `C` 写回局部性。
- 计算 kernel 仍没有 IO/Compute 双缓冲、SRAM 或 Memory Core 流水。

## 8. 各 Kernel 实现对比

| Kernel | 每行内层迭代 | metadata 地址 | NRAM 主缓冲 | 当前流水线 | 输出寻址 |
| --- | --- | --- | --- | --- | --- |
| CSR | `crow[row + 1] - crow[row]` | `ptr++` | 双 accumulator + 双 `B` | 非零元内层流水 + output tile 外层流水 | `row * c_row_stride` |
| ELL | 固定 `ell_width` | `row * ell_width + slot` | accumulator + `B` | 同步 Load/Compute | `row * c_row_stride` |
| SELL | 当前 `slice_width` | `slice_begin + slot * slice_height + row_in_slice` | accumulator + `B` | 同步 Load/Compute | `storage_row * c_row_stride` |
| SELL-sigma-c | 当前 `slice_width` | 与 SELL 相同，额外读取 `row_indices` | accumulator + `B` | 同步 Load/Compute | `row_indices[storage_row] * c_row_stride` |

## 9. CSR 后续优化路线

本节按优先级给出建议。优先级考虑通用性、预期收益、实现风险、热路径复杂度和片上存储代价。

### P0：建立可解释的性能基线和 dispatch 策略

这是后续所有优化的前提。

#### 9.1 增加 CSR 专项基准矩阵

至少覆盖：

- `N = 1, 2, 8, 32, 128, 512, 4096, 25000+`。
- 平均行 nnz 从 0、1、2 到数百。
- 均匀行长。
- 包含大量空行。
- 单个或少数超长行。
- power-law/图邻接矩阵分布。
- 随机 `col_indices`、有序列索引、带状列索引和重复列索引。
- `int32` 与 `int64`。
- `beta = 0`、`beta = 1` 和一般值。
- 连续与非连续 `B/C`。

专项 kernel 用例应包含非空行与空行交替的宽矩阵，并同时覆盖 `beta == 0` 和 `beta != 0`，以触发 NRAM 多 tile、尾块、同一 task 的 accumulator 复用以及空行的延迟 Store 路径。

#### 9.2 同时记录算法指标和实际流量

建议记录：

- latency。
- theory ops，主体约为 `2 * nnz * N`，并单独计算 `alpha/beta` 开销。
- theory IO size。
- 实际 packed metadata bytes。
- 实际 `B` load bytes，包括不同输出行重复引用同一 `k` 时的重复读取。
- Compute efficiency 和 IO efficiency。
- 每行 nnz 方差、最大行 nnz、空行比例。
- profiler 中 IO/Compute/Scalar 流忙碌比例和同步等待。

对于 ELL/SELL，应同时报告逻辑 nnz 和 packed nnz，否则无法解释 padding 带来的性能差异。

#### 9.3 增加流水线阈值

当前所有连续列且 `N > 1` 的 CSR 都使用四缓冲组合流水线。建议保留一个简单的非流水 NRAM kernel，并根据以下特征选择：

- `N` 或当前 tile 长度。
- 平均行 nnz。
- 最大行 nnz。
- CSR column tile 数量。
- 索引类型。

当向量过短时，每个非零元的同步成本可能高于 IO/Compute 重叠收益。此时两缓冲非流水 kernel 还可以使用更大的 column tile。

### P1：降低 CSR 元数据和同步开销

#### 9.4 分块搬运 `values` 和 `col_indices`

将一段 CSR 行的 `values` 和 `col_indices` 以连续 DMA 搬入 NRAM，再在 NRAM 中逐项消费：

```text
GDRAM values/indices -> NRAM metadata tile
GDRAM B rows         -> NRAM B ping/pong
Compute              -> NRAM accumulator
```

收益：

- 将逐项小粒度 GDRAM load 变成连续大块搬运。
- 减少 Scalar 流等待 metadata。
- 避免 metadata load 与下一块 `B` 大搬运互相干扰。

代价：

- 需要从 240 KiB NRAM 中再划分 metadata buffer。
- `int64` metadata 占用更大。
- metadata tile 大小需要与 column tile 联合调优。

可以先实现小型 metadata tile，例如一次缓存 32 或 64 个非零元，再依据 profile 调整。

#### 9.5 以非零元小批次摊销同步

当前每个非零元执行 pipeline 同步。可以把多个非零元组织成一个软件 stage：

- 一次预取若干 `B` 子块或 metadata。
- 使用更深的 ping/pong ring。
- 每批而不是每个非零元同步。

需要谨慎控制 NRAM 占用。若 `N` 很宽，单个 `B` tile 已经很大，批量缓存多个完整 `B` 行块不现实；可以将 `N` tile 进一步缩小，换取更深的 pipeline，然后用实测选择平衡点。

#### 9.6 常见 `alpha/beta` 特化

建议至少提供：

- `alpha == 1 && beta == 0`。
- `alpha == 1 && beta == 1`。
- `beta == 0`。
- 一般 `alpha/beta`。

模板特化可以去掉热路径中的分支、旧 `C` 读取和部分向量指令。

#### 9.7 检查 fused AXPY/FMA 能力

在目标 Neuware/BANG C SDK 上确认是否存在适合以下计算的向量或张量 primitive：

```text
accumulator = accumulator + sparse_value * dense_vector
```

如果存在，可以替换当前 `mul_scalar + add`，减少 NRAM 临时写回和 Compute 指令数。如果不存在，应保留当前实现，不要用标量循环模拟融合。

### P1：解决 CSR 行负载不均衡

对通用 CSR，负载均衡通常比继续增加单 core pipeline 深度更重要。

#### 9.8 nnz-aware 静态调度

当前按 output tile 数量均分。可以在显式预处理阶段或首次执行时通过 device kernel 计算每行成本：

```text
row_cost = row_nnz * number_of_column_tiles
```

再把 output tiles 分配到成本近似相等的 task 区间。该方案保持每个 tile 单 owner，不需要原子操作。

实现选择包括：

- 按行 nnz 分桶。
- 构建重排行号数组，但写回仍使用原始行号。
- 为不同 nnz bucket 启动不同 kernel。

nnz-aware 调度表可以由独立的 device 预处理 kernel 写入临时 GDRAM task list。稀疏矩阵重复执行时应复用该 task list；性能报告需要分别给出预处理开销和稳态 kernel latency。

#### 9.9 长行拆分

当少数行远长于平均值时，一个 task 遍历整行会形成尾部延迟。可将长行沿 nnz 维拆给多个 task：

```text
partial[task, row, col_tile] = partial SpMM
C[row, col_tile] = reduce(partials)
```

归约方式可以是：

- Cluster SRAM 内归约，适合一个 Cluster 协作处理长行。
- GDRAM partial buffer 加第二个 reduction kernel。
- 原子累加，仅在冲突和数值非确定性可接受时使用。

短行继续使用单 owner kernel，避免所有行都承担归约成本。最终应形成 hybrid policy。

#### 9.10 动态工作队列

另一种方案是用全局或 Cluster 级计数器动态领取 `(row, col_tile)`。它能处理不可预测的行长分布，但会引入原子计数器、调度和地址读取开销。

建议先实现静态 nnz-aware 调度；只有静态分桶仍不能解决的极端分布再考虑动态队列。

### P2：扩大 Load/Compute/Store 流水覆盖

当前已经使用：

```text
acc_ping + acc_pong + B_ping + B_pong
```

它通过下一 tile 首块 lookahead，让 `C[t]` Store 与 `tile[t + 1]` 的首段 Compute 重叠。进一步优化的目标是扩大可重叠区间，而不是再次增加同样的双缓冲。

当前限制是 Load 和 Store 共用 IO 流。可以研究：

- 在 Store 前预取一小批下一 tile 的 metadata 和多个较小 `B` 子块，使下一 tile 有更长的独立 Compute 区间。
- 将完成的 accumulator 先搬到 SRAM，再由 Memory Core 写回。
- 用独立 kernel 或临时 GDRAM buffer 延迟批量写回，但要计算额外 GDRAM 流量。
- 对只有一个 tile、空行或很短向量的场景回退到较浅流水线。

需要比较“扩大流水覆盖”与“更小 tile、更多 metadata buffer 及 B/C DMA 更碎片化”的净收益。

### P2：Memory Core + SRAM 集群级流水

目标是同时使用：

```text
Memory Core IO:  GDRAM -> SRAM[next]
MLU Core Move:           SRAM[curr] -> NRAM[next]
MLU Core Compute:                      compute NRAM[curr]
Output path:                                      store previous result
```

需要 SRAM ping/pong 和 NRAM ping/pong，并使用 Cluster 级同步保证 Memory Core 与 MLU Core 的生产者/消费者关系。

CSR 的困难在于 `col_indices` 使 `B` 行随机。建议的任务重构是：

1. 一个 Cluster 负责一个输出行或一小组输出行。
2. 四个 MLU Core 沿 `N` 维分列处理。
3. Memory Core 根据一批 CSR `col_indices`，将对应 `B` 行的不同列分块搬入 SRAM 分区。
4. MLU Core 使用 Move 流从 SRAM 搬入私有 NRAM。
5. Compute 流处理上一 stage。

该路径更适合：

- `N` 较宽。
- 每行 nnz 足够多。
- 一个 Cluster 有足够连续工作来摊销同步。
- `B` 行块搬运是主要瓶颈。

它不一定适合短行、小 `N` 或高度随机且小粒度的访问。必须保留当前 Core-private kernel 作为 fallback。

### P2：改善 `B` 局部性和复用

#### 9.11 排序列索引与重复列合并

若 CSR 每行 `col_indices` 有序：

- 地址预测和预取更稳定。
- 重复列可以先合并 `values`，只加载一次 `B` 行。
- 相邻 `B` 行可能改善 GDRAM/L2 局部性。

但排序和重复项合并会增加预处理成本，并可能改变浮点累加顺序。应作为可选预处理，仅在稀疏矩阵重复使用时启用。

#### 9.12 小型 `B` 行缓存

图计算等场景中，相邻 CSR 行可能引用相同 `B` 行。可以研究：

- Cluster SRAM 中的软件 cache。
- 对一组输出行进行 column-index 分桶。
- 缓存最近使用的少量 `B` tile。

缓存命中率不足时，tag 检查和 SRAM 占用会使性能下降，因此必须基于真实模型矩阵统计，而不是默认启用。

### P2：通用 stride 快路径

对于 `col_stride != 1`：

1. 将 strided `B` gather 到连续 NRAM。
2. 在 NRAM 中执行向量计算。
3. 将结果 scatter 到 strided `C`。

也可以使用临时 GDRAM buffer 将 `B` 或 `C` 转换为连续布局。选择依据是：

- 稀疏计算量是否足以摊销布局转换。
- `B` 是否会被多个 SpMM 重复使用。
- stride 模式是否仍能形成较大的连续 GDRAM chunk。

### P3：块结构专用 Kernel

当 CSR 的列索引呈现稳定块结构时，可以增加 BSR 风格 kernel，将多个相邻非零元组织成小矩阵块，并研究使用 MLU 张量指令。该路径需要独立的块布局和任务分解，不应增加通用 CSR kernel 热路径中的判断。

这是高收益但高复杂度方向，应在 CSR 基础 kernel、负载均衡和 benchmark 完成后推进。

## 10. ELL/SELL 系列 Kernel 优化方向

### 10.1 ELL

优先方向：

1. 为连续列 kernel 增加与 CSR 类似的 B-load/Compute ping-pong，但使用独立阈值，避免短向量同步开销。
2. 增加显式 row length 或 padding sentinel，减少每个 slot 的 `value == 0` 判断。
3. 研究 column-major ELL 或多行协同 kernel，让相邻 core 在同一 slot 上读取连续 metadata。
4. 自动估算 padding ratio；超过阈值时回退 CSR 或使用 ELL+CSR/COO hybrid，将长行放入 overflow 格式。
5. 根据 `ell_width` 和 `N` 选择按行、按列 tile 或一个 Cluster 多行协作。

### 10.2 SELL

优先方向：

1. 根据 MLU Core/Cluster 结构自动调优 `slice_height`。
2. 让一个 Cluster 显式处理一个或多个 slice，使同一 slot 的 metadata 访问和 task 生命周期更一致。
3. 使用 SRAM staging 同一 slice 的 `values/col_indices`。
4. 增加 B ping-pong 和 metadata tile。
5. 保存每个 storage row 的真实长度，避免遍历 padding slot。
6. 根据 slice padding ratio 对异常长行使用 CSR overflow。

### 10.3 SELL-sigma-c

优先方向：

1. 联合搜索 `sigma` 和 `slice_height`，目标函数同时考虑 padding、排序成本和 `C` 写回局部性。
2. 比较 fused scatter 与“先连续写 permuted `C`、再启动 reorder kernel”两种输出策略。
3. 增加与 SELL 相同的 slice-level Cluster 协作和流水线。
4. 分别报告排序预处理和纯计算 kernel 的耗时。

## 11. 推荐实施顺序

建议按以下顺序继续优化 MLU SpMM：

1. 建立 CSR 专项 benchmark 和 profiler 数据，保留当前 kernel 作为 baseline。
2. 增加 pipeline/non-pipeline 自适应 dispatch，解决短向量退化。
3. 分块搬运 CSR `values/col_indices`，降低 scalar metadata 开销。
4. 实现 nnz-aware 静态调度和长行 hybrid 拆分，解决最主要的通用 CSR 负载不均衡。
5. 尝试常见 `alpha/beta` 特化和 fused AXPY/FMA primitive。
6. 在宽 `N`、长行 workload 上扩大现有 Load/Compute/Store 流水的重叠区间。
7. 实现 Memory Core + SRAM 的 Cluster 协作 CSR kernel，并通过 dispatch 只覆盖适合场景。
8. 将有效的流水和调度策略推广到 ELL/SELL/SELL-sigma-c。
9. 最后研究 `B` cache 和块稀疏张量指令路径。

这个顺序优先解决 CSR 的通用瓶颈，避免在没有测量数据时直接增加 SRAM、多级同步和更多缓冲区。

## 12. Kernel 正确性和性能验收

### 12.1 正确性

MLU kernel 测试至少应覆盖：

- CSR、ELL、SELL 和 SELL-sigma-c 的各条 kernel 路径。
- `int32/int64` 索引。
- 空行、全空矩阵、单非零元行和超长行。
- `N == 1`、对齐 tile、非对齐尾 tile、多 column tile。
- `alpha = 0/1/general`。
- `beta = 0/1/general`。
- 连续和非连续 stride。
- ELL/SELL padding。
- SELL-sigma-c 非平凡 row permutation。
- 重复列索引和真实零值。

在 Neuware 环境中还应使用可用的 BANG memcheck 工具检查异步搬运、尾块和 ping-pong 生命周期。

### 12.2 性能

每项优化至少报告：

- 优化前后 latency。
- 矩阵形状、nnz、行长统计、索引类型和格式参数。
- 纯计算 kernel latency；存在预处理 kernel 时单独报告其 latency。
- theory ops 和 theory IO。
- IO/Compute efficiency。
- profiler 中 IO、Compute、Move、Scalar 流利用率。
- 同步等待比例。
- NRAM/SRAM 使用量和 tile 大小。

## 13. 维护约束

后续修改应保持：

- 对齐计算长度与有效写回长度分离。
- ping/pong 被 IO 覆盖前，前一次 Compute 已完成。
- accumulator 被下一 tile 清零前，前一次输出 IO 已完成。
- 一个输出 tile 只有一个 owner，除非新路径明确实现归约或原子策略。
- 通用 stride 要么正确向量 gather/scatter，要么明确保留 scalar fallback。
- 所有性能优化保留可比较的 baseline kernel 和覆盖边界条件的测试。
