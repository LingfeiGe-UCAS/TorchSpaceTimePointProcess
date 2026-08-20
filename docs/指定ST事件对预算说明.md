# 指定 S×T 事件对预算

## 目标

`DistributedExactLikelihood` 支持可选的 `pair_budget`：

```python
DistributedExactLikelihood(
    exact,
    pair_budget=16_000_000,
    source_alignment=256,
)
```

`pair_budget` 是单个似然计算块允许的最大事件对元素数：

```text
target_chunk_size × source_chunk_size <= pair_budget
```

不指定 `pair_budget` 时，程序保留原来的 block-cyclic 行为。

## 分配方式

1. 目录继续保持非递减时间顺序。
2. 每个 target 的工作量估计为“全部 KDE 中心数 + 严格更早的触发 source 数”。
3. 按累计工作量把 target 划分为与进程数相同的连续区间。
4. 触发补偿项的 source 按数量划分为连续且互斥的区间。
5. 使用最大的本地 target 数和 `pair_budget` 推导所有 rank 共用的 source chunk。
6. 每个 target 仍与完整 source 历史计算，`delta_time > 0` 的因果规则不变。

## Noto / L20 起始配置

float64、checkpoint 开启时，可以先使用 4000×4000 预算：

```text
pair_budget = 16000000
source_alignment = 256
```

四卡示例：

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/distributed_exact_likelihood.py \
  configs/noto_ityp0_etas3d_fortran_em.toml \
  --pair-budget 16000000 \
  --source-alignment 256
```

六卡只需把 `--nproc-per-node` 改为 6。程序输出的
`pair_budget_plan` 会记录实际 target/source 分片和推导出的计算 chunks。

## 当前范围

本阶段只改进精确似然 M 步的分配。E 步仍使用原实现。

当前实现要求 `pair_budget` 不小于实际采用的 target chunk；否则会报错。
