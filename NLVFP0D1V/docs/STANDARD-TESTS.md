# NLVFP0D1V 标准测试

所有输入来自解析初态或制造问题，无随机种子。运行 Pkg.test 可执行完整集合，smoke_kinetic.jl 可单独执行两个物理算例。测试过程写控制台，双输入冲突测试的临时文件由 Julia 管理。

| 测试 | 文件 | 判据 |
| --- | --- | --- |
| 径向测度与观测 | [runtests.jl](../test/runtests.jl) | 体积权重、带符号源、移动权函数、基矩阵复制、矩阶和归一化防误用 |
| 独立 King 积分 | [radial_moments.jl](../test/radial_moments.jl) | 有径向偏置的两核混合，直接径向求积对比偶阶、奇阶及连续整数阶解析矩，门限 \(3\times10^{-12}\) |
| 时间阶和切换 | [evolution.jl](../test/evolution.jl) | 制造解的相邻两次步长减半误差比在 3.95–4.05，检查切换点左右值，超门触发拒绝 |
| 双文件输入 | [case_inputs.jl](../test/case_inputs.jl) | 核对尺度，拒绝同一文件复用、交叉算例与不一致尺度 |
| 解析等温平衡 | [物理](../test/cases/ed_equilibrium/physical-input.toml) · [无量纲](../test/cases/ed_equilibrium/dimensionless-input.toml) | 电子、氘均为 12.5 keV，检查净热交换接近零，温度、守恒、总 H 与状态稳定 |
| 非线性热弛豫 | [物理](../test/cases/ed_relaxation/physical-input.toml) · [无量纲](../test/cases/ed_relaxation/dimensionless-input.toml) | 电子 15 keV、氘 10 keV，自碰撞和互碰撞同时作用。初态核对解析热交换率，跨共宽、固定仿射、变斜率核对热流方向、守恒、H 与 CPE |
| 精度与版本 | [execution.jl](../test/execution.jl) · [verify_freeze.jl](../scripts/verify_freeze.jl) | 分层精度与回算误差、版本一致性、摘要和继承源码哈希 |

热弛豫和等温测试只走两个 \(2^{-31}\tau\) 微步。逐物种粒子数偏差和总动能偏差采用各自理论值归一，统一门限 \(2\times10^{-12}\)。总动量结构零，单步 \(H\) 增量门为 \(10^{-11}\)，硬矩回算门为 \(2\times10^{-12}\)，缩放后的无穷范数隐式残差门为 \(10^{-12}\)。短窗测试支持程序定版，完整的分布、温度与高阶矩对照在后续统一计算时验收。
