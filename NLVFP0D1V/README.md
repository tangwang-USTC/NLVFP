# NLVFP0D1V

均匀、各向同性的非线性 Vlasov–Fokker–Planck（NLVFP）径向程序库。0D1V 表示无物理空间变化，且速度分布只有一个径向坐标 \(v=|\mathbf v|\)。径向泛函采用三维速度测度 \(4\pi v^2\,dv\)。

当前项目内定版以 [Project.toml](Project.toml) 为准。当前求解器覆盖电子–氘自碰撞、互碰撞、共宽和仿射异宽 King 闭合。适用范围及精度见 [架构与数值合同](docs/ARCHITECTURE.md)。

## 安装、测试与校验

从本程序库目录执行。依赖锁使用 Julia 1.12.6 生成。

```powershell
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --threads=1 --project=. -e 'using Pkg; Pkg.test(; allow_reresolve=false)'
julia --startup-file=no --project=. scripts/verify_freeze.jl
```

可从程序库目录单独运行某一组。

```powershell
julia --startup-file=no --threads=1 --project=. scripts/smoke_kinetic.jl
julia --startup-file=no --threads=1 --project=. scripts/benchmark_projection.jl
julia --startup-file=no --threads=1 --project=. scripts/benchmark_kinetic.jl
```

## 最小用法

从两个输入文件加载同一个算例。物理文件给物种参数和温标，无量纲文件给速度域、时间网格，并核对派生尺度。例子只走标准测试的两个微步。

```julia
using NLVFP0D1V

deck = load_case("test/cases/ed_relaxation/physical-input.toml",
                 "test/cases/ed_relaxation/dimensionless-input.toml")
model = king_case1(deck; width_rule=:affine, slope=constant_slope())
result = evolve!(StepWorkspace(), model, model.initial, deck.times_tau)
@show result.diagnostic result.residual
```

## 目录

| 位置 | 用途 |
| --- | --- |
| src | 公共入口、观测与矩合同、输入解析、King 闭合与时间推进 |
| src/internal | 定版所需的项目内来源源码快照和出处哈希 |
| test | 程序回归、解析与制造问题、独立求积、双输入标准算例 |
| scripts | 单独开展物理冒烟、性能基准和定版校验 |
| docs | 数学范围、尺度、接口和使用说明 |
| versions | 项目内版本冻结清单 |

[标准测试说明](docs/STANDARD-TESTS.md) · [版本维护约定](CHANGELOG.md) · [0.0.1 冻结清单](versions/0.0.1.toml)
