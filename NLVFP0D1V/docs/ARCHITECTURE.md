# NLVFP0D1V 架构与数值合同

当前版本为 0.0.1。版本权威见 [Project.toml](../Project.toml)。

## 方程与适用范围

速度变量 \(v=|\mathbf v|\ge0\)，求解各向同性非线性 Landau 碰撞方程的径向弱式
\[
 \partial_t f_a(v)=\sum_{b\in\{e,D\}}C^{\mathrm L}_{ab}[f_a,f_b](v),
 \qquad
 n_a=4\pi\int_0^\infty f_a(v)v^2\,dv.
\]
每步从两物种当步分布计算 \(C^{\mathrm L}_{ab}\)，保留自碰撞和成对互碰撞。均匀模型中空间输运消失，边界为径向原点的正则性及热尾衰减，数值积分域由各物种的速度节点和右端给定。由于各向同性，总矢量动量为结构零。本版本的求解入口面向 e–D 双物种的两 Maxwellian 解析初态；电磁力、各向异性和空间非均匀演化属于其他模型。

电子用 42 个广义观测的正对数表示，氘用十个 King 与至 \(M_{20}\) 的连续偶阶硬矩。在同一时刻解壳矩正反演，重构分布供碰撞率求值。尺度化宽度、偏置关系为 \(s_i=1-d+\lambda q_i\)，公共斜率取零对应共宽。仿射默认 0.005，可由非负时间律、检查点律或状态律给定。将斜率作为额外未知量反演或为其给独立方程，需要增广当前状态和阶段残差。

## 输入、尺度和诊断

load_case 采用 schema 1 的物理权威模式，核对两个独立 TOML 的物种顺序、算例 ID、径向模型、来源和七项数值断言。缺失必要尺度或断言冲突会终止装载。物理文件使用 SI 的质量、粒子数密度和电荷数，温度用 keV 的能量单位。离散输入给电子和氘各自热速度单位上的速度域、源求积阶和 \(t/\tau\) 节点。

取 \(m_*=m_e\)、\(n_*=n_e(0)\)、参考温度能量 \(E_*=T_*\) 和 \(v_*=\sqrt{2E_*/m_*}\)。对 \(\hat m=m/m_*\)、\(\hat n=n/n_*\)、\(\hat T=T/E_*\)，输入派生的 \(\hat v_{\rm th}=\sqrt{\hat T/\hat m}\)，\(\hat f=fv_*^3/n_*\)，因此 \(\int\hat f\,4\pi\hat v^2d\hat v=\hat n\)。初态总能量在 \(m_*n_*v_*^2\) 单位下为 \(3\sum_a\hat n_a\hat T_a/4\)。令 \(\hat T_{\rm eq}=\sum_a\hat n_a\hat T_a/\sum_a\hat n_a\)。程序采用的参考碰撞系数和时间尺度为
\[
 \Gamma_*=\frac{3\sqrt{\pi}\,\hat m_e\hat m_D
          [\hat T_{\rm eq}(\hat m_e^{-1}+\hat m_D^{-1})]^{3/2}}
          {16(\hat n_e+\hat n_D)},\qquad
 \tau=\frac{\Gamma_*m_*^2v_*^3}{n_*\Gamma^{\rm SI}_{eD}},
\]
其中 \(\Gamma^{\rm SI}_{ab}=2\pi[e^2/(4\pi\epsilon_0)]^2
(Z_aZ_b)^2\ln\Lambda_{ab}\)，代码中的 \(\hat\Gamma_{ab}=\Gamma_*\Gamma^{\rm SI}_{ab}/\Gamma^{\rm SI}_{eD}\)。时间和速度分别乘 \(\tau\)、\(v_*\) 恢复 SI，分布乘 \(n_*/v_*^3\) 恢复数密度。尺度返回在 deck.scales 中，数值门限针对无量纲方程。

speed_moment_target 的 :thermal_maxwell 保存相对含粒子数的同温 Maxwell 矩之比。shell_targets 要求连续偶阶 \(M_0,M_2,\ldots\) 和 \(M_0=M_2=1\)。预设偶阶、奇阶、连续整数阶及自定义径向权函数都由同一个投影框架处理。\(H=\sum_a\int \hat f_a\log\hat f_a\,d^3\hat v\) 用于离散下降检查，物理熵符号相反。

## 程序接口与精度

| 组件 | 稳定调用点 |
| --- | --- |
| 两文件输入 | load_case，随后用 king_case1 创建问题 |
| 观测投影与读回 | radial_measure、radial_space、power_space、observe、observe!、observe_rate、king_observations |
| 静态闭合 | speed_moment_target、shell_targets、invert_zero_offset |
| 闭合策略与动态求解 | constant_slope、linear_time_slope、checkpoint_slope、functional_slope、StepWorkspace、StepOptions、advance_step!、evolve! |
| 执行口径 | execution_metadata。返回库版本、环境哈希、线程数、精度和门限 |

运行态、隐式迭代以及特征分解用 IEEE binary64，天工标识 p64，64 位存储、53 位有效尾数、epsilon 为 \(2^{-52}\)。仿射矩阵前处理和正壳矩 Cholesky 分别使用 MPFR 的 256 和 160 位有效尾数，随后转换 Float64 完成特征分解。MPFR 有效尾数与对象存储位宽分别理解。零偏置正反演额外报告其指定的构造精度及回算残差。秩、正性和残差门必须同时检查，测试通过的范围见 [标准测试](STANDARD-TESTS.md)。

默认在 \(0.125\tau\) 前用 SDIRK2，之后用隐式梯形；使用调用方给定的时间网格并对齐斜率节点。隐式阶段使用阶段时间和状态取斜率。分段常值策略在切换点核对连续分布、热源与总 \(H\)，失败回到调用者持有的父态。接受步检查粒子数、总动量、总能量、\(H\) 及硬矩残差。

可复用源码随本包固定在 src/internal，原始路径、提交与文件 SHA-256 见 [来源清单](../src/internal/PROVENANCE.toml)。入口在复制出项目的独立目录也可加载，只依赖 Project.toml 和 Manifest.toml 声明的 Julia 包。
