# MKLTwoStageRO.jl
`MKLTwoStageRO.jl` is a Julia package for multiple kernel learning (MKL) aided two-stage robust optimization (TwoStageRO). 

Generally, it solves the following TwoStageRO problem:

<!-- $$
\begin{aligned}
  &{\min_{\mathbf{x}} \mathbf{a}^\top\mathbf{x} + \max_{\mathbf{u}\in\mathcal{U}}\min_{\mathbf{y}\in\mathcal{Y}(\mathbf{x},\mathbf{u})}\mathbf{b}^\top\mathbf{y}}\\
  &{\begin{aligned}
    \mathrm{\,s.t.}\qquad&\mathbf{Ex}\geq\mathbf{e},\quad\mathbf{x}\in\mathcal{S}_{\mathbf{x}}\\
    &\mathcal{Y}(\mathbf{x}, \mathbf{u}) = \{\mathbf{y}\in\mathcal{S}_{\mathbf{y}}: \mathbf{Ax}+\mathbf{By}+\mathbf{Cu}\geq\mathbf{c}\}\\
    &\text{with } \mathcal{S}_{\mathbf{x}} \subseteq\mathbb{R}^p_+\text{, }\mathcal{S}_{\mathbf{y}} \subseteq\mathbb{R}^q_+\text{ and }\mathcal{U}\subseteq\mathbb{R}^r \text{ the uncertainty set.}  
  \end{aligned}}
\end{aligned}
$$ -->
<span id="F1">
<p align=center>
  <img src="/assets/TwoStageRO.png" width=500>
</p>

It includes data-driven MKL uncertain set construction [[1](#R1), [2](#R2)], and algorithms like MKL uncertainty set-induced column-and-constraint generation (MKLCCG) [[2](#R2)], as well as extended column-and-constraint generation (ECCG) [[3](#R3)], original (nested) column-and-constraint generation (CCG) [[4](#R4), [5](#R5)], and Benders-dual cutting plane method (BDCP) [[4](#R4)].


## Package Installation

``` julia
using Pkg
Pkg.add("MKLTwoStageRO")
Pkg.add("GLMakie") # if the user wants visualization
```

Then, `using` the necessary packages:
``` julia
using MKLTwoStageRO
using GLMakie
```

## A Toy Example: Two-Stage Robust Location-Transportation Problem

Consider the following location-transportation problem from [[4](#R4)]. To supply a commodity to customers, it will be first to stored at $m$ potential facilities and then be transported to $n$ customers. The fixed cost of the building facilities at site $i$ is $f_i$ and the  unit capacity cost is $a_i$ for $i=1,\cdots,m$. The demand is $d_j$ for $j=1,\cdots,n$, and the unit transportation cost between $i$ and $j$ is $c_{ij}$ for $i$ - $j$ pair. The maximal allowable capacity of the facility at site $i$ is $K_i$ and $\sum_iK_i\geq\sum_jd_j$ ensures feasibility. Let $y_i\in\\{0,1\\}$ be the facility location variable, $z_i\in\mathbb{R}\_+$ be the capacity variable, and $x_{ij}\in\mathbb{R}\_+$ be the transportation variable. 

In practice, the demand is uncertain before any facility is built and capacity is installed. However, in many cases, we can obtain some demand data (historical or experimental) $\mathcal{D}$ through market research and other methods. The question is how to make robust decisions based on these data.

This problem can be modeled as the following two-stage robust optimization:

<!-- $$
\begin{aligned}
  &{\min_{\mathbf{y}\in\{0,1\}^m,\mathbf{z}\in\mathbb{R}^m_+  } \sum_{i}(f_iy_i+a_iz_i) + \max_{\mathbf{d}\in\mathcal{U}(\mathcal{D} )}\min_{\mathbf{x}\in\mathbb{R}^{m\times n}_+ }\sum_{i,j}c_{ij}x_{ij} }\\
  &{\begin{aligned}
    \mathrm{\quad\;\;\; s.t.}\qquad& z_i\leq K_iy_i,\quad\forall i\\
    &\sum_j x_{ij}\leq z_i,\quad\forall i\\
    &\sum_i x_{ij}\geq d_j,\quad\forall j\\
    &\text{with } \mathcal{U}(\mathcal{D} ) \text{ the data-driven uncertainty set.}  
  \end{aligned}}
\end{aligned}
$$ -->
<p align=center>
  <img src="/assets/RobustLocationTransportation.png" width=450>
</p>

Here we instantiate the above problem with the following parameters:
``` julia
f = [400, 414, 326]
a = [18, 25, 20]
c = [22 33; 33 23; 20 25]
K = [5, 5, 5]
```


### Collecting uncertain data
Assume the following is the demand data we have collected. Note that each column of the training data corresponds to an observation of the input features.
``` julia
d1 = 0.5 .+ 4 * rand(300)
d2 = 2 ./ d1 + 0.3 * randn(300) .+ 1
D = [d1'; d2']
mklocsvmplot(D; backend=GLMakie) # visualize the demand data
```

<p align=center>
  <img src="/assets/DemandData.png" width=70%>
</p>

### Constructing MKL-based uncertainty set

``` julia
algor = HessianMKL(verbose=false)
U = mklocsvmtrain(D, 21; kernel="DNPNK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05)
mklocsvmplot(U; backend=GLMakie) # visualize the MKL uncertainty set
```
<p align=center>
  <img src="/assets/UncertaintySet.png" width=70%>
</p>

Note:
1. `kernel`: Generally, the user has two choices, `"DPDK"` or `"DNPNK"`.
2. `algorithm`: An instance of `HessianMKL` or `QCQP`.
3. `q_mode`: `"evenly"` or `"randomly"`.
4. `ν`: It stands for an upper bound on the proportion of the outlies and hence controls the conservatism of the robust optimization.

Please see [[1](#R1),[2](#R2)] and package [`MKLOneClassSVM.jl`](https://github.com/hanb16/MKLOneClassSVM.jl) for more information.



### Modeling the problem

The user needs to organize the model into the [general form](#F1) and specify each component in it, thus establishing a `TSROModel`.

``` julia
a = [400, 414, 326, 18, 25, 20]
b = [22, 33, 20, 33, 23, 25]
E = [5 0 0 -1 0 0; 0 5 0 0 -1 0; 0 0 5 0 0 -1]
e = [0.0, 0.0, 0.0]
B = [-1.0 0.0 0.0 -1.0 0.0 0.0; 0.0 -1.0 0.0 0.0 -1.0 0.0; 0.0 0.0 -1.0 0.0 0.0 -1.0; 1.0 1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 1.0 1.0]
c = [0.0, 0.0, 0.0, 0.0, 0.0]
A = [0.0 0.0 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0]
C = [0.0 0.0; 0.0 0.0; 0.0 0.0; -1.0 0.0; 0.0 -1.0]
Sx = [ZeroOne(3), Rplus(3)]
Sy = [Rplus(6)]
model = TSROModel(a, b, E, e, B, c, A, C, Sx, Sy, U)
```



### Solving the model using MKLCCG algorithm

``` julia
mklccg = MKLCCG()
x_star, objv, recourse = solve(model; algorithm=mklccg)
```



### Retrieving the recourse decision

``` julia
û = [2.0, 3.0] # assume this is the uncertainty observation
y_star, RPobjv = recourse(û)
```


## Other Features

### The MKL uncertainty set can be trained distributedly

``` julia
using Distributed
addprocs(3)
@everywhere using MKLTwoStageRO

Û = mklocsvmtrain(D, 21; kernel="DNPNK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05, num_batch=3)
mklocsvmplot(Û; backend=GLMakie) # visualize the MKL uncertainty set
```

<p align=center>
  <img src="/assets/UncertaintySet_dis.png" width=70%>
</p>


Then we can also build the `TSROModel` and solve it:
``` julia
model = TSROModel(a, b, E, e, B, c, A, C, Sx, Sy, Û)
x_star, objv, recourse = solve(model; algorithm=mklccg)
```




### Other algorithms can also be used

If there is an MKL-based uncertainty set `U`, the user doesn't have to use MKLCCG algorithm to solve the `model` (although it is recommended). There are other two-stage robust optimization algorithms available in this package for selection, such as the previously mentioned extended column-and-constraint generation (ECCG) [[3](#R3)], original (nested) column-and-constraint generation (CCG) [[4](#R4), [5](#R5)], and Benders-dual cutting plane method (BDCP) [[4](#R4)]. However, before calling these algorithms, it is necessary to first convert the MKL uncertain set into a `JuMP` model expression and then build the `model` again, e.g.,

``` julia
Ū = convert_to_jumpmodel(U; form="linear", varname=:u)
model = TSROModel(a, b, E, e, B, c, A, C, Sx, Sy, Ū)
eccg = ECCG()
x_star, objv, recourse = solve(model; algorithm=eccg)
```

Note that different algorithms currently support slightly different types of problems, roughly as follows:
- `MKLCCG`: `model.U` should be an MKL-based uncertainty set. `model.Sx` is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space, and `model.Sy` should be a nonnegative real-valued space. The **Feasibility** assumption is necessary.
- `ECCG`: `model.U` can be modeled as a `model` from [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) or a `Polyhedron` from [`Polyhedra.jl`](https://github.com/JuliaPolyhedra/Polyhedra.jl). `model.Sx` is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space, and `model.Sy` should be a nonnegative real-valued space. The **Feasibility** assumption is necessary.
- `CCG`: `model.U` can be modeled as a `model` from [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) or a `Polyhedron` from [`Polyhedra.jl`](https://github.com/JuliaPolyhedra/Polyhedra.jl). `model.Sx` and `model.Sy` are allowed to be nonnegative real-valued spaces or nonnegative mixed integer spaces. The **Relatively Complete Recourse** assumption is necessary when `model.Sy` is a nonnegative real-valued space, and the **Extended Relatively Complete Recourse** assumption is necessary when `model.Sy` is a nonnegative mixed integer space, and `model.Sy` need have at least one real-valued dimension.
- `BDCP`:  `model.U` can be modeled as a `model` from [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) or a `Polyhedron` from [`Polyhedra.jl`](https://github.com/JuliaPolyhedra/Polyhedra.jl). `model.Sx` is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space, and `model.Sy` should be a nonnegative real-valued space. The **Relatively Complete Recourse** assumption should hold.



### The uncertainty set doesn't have to be an MKL-based one


The user can also construct uncertainty sets using the syntax of [`JuMP.jl`](https://github.com/jump-dev/JuMP.jl) or [`Polyhedra.jl`](https://github.com/JuliaPolyhedra/Polyhedra.jl), for example:

- Case-1: 
``` julia
c = [1, 2]; b = [3];
A = [4 5]; d = [6]; 
G = [1;;]; h = [2]; E = [-3 -4]; M = [-5 -6];
Sx = [Rplus()]; Sy = [Rplus(), ZeroOne()];
UncMod = Model()
u0 = [0, 0]
@variable(UncMod, u[1:2])
@variable(UncMod, -1 <= δ[1:2] <= 1)
@constraint(UncMod, u == u0 + δ)
model = TSROModel(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
ccg = CCG()
y_star, objv, recourse = solve(model; algorithm=ccg)
û = [0.5, -0.5]
x_star, RPobjv = recourse(û)
```

- Case-2: 
``` julia
c = [2]; b = [1];
A = [0;;]; d = [0]; 
G = [1;;]; h = [0]; E = [-1;;]; M = [-1;;];
Sy = [ZeroOne()]; Sx = [Rplus()]; 
UncSet = polyhedron(HalfSpace([1], 1) ∩ HalfSpace([-1], 0)) # 0 <= u <= 1
model = TSROModel(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
bdcp = BDCP()
y_star, objv, recourse = solve(model; algorithm=bdcp)
û = [0.5]
x_star, RPobjv = recourse(û)
```


## Citing
If you find `MKLTwoStageRO.jl` useful, we kindly request that you cite this [repository](https://github.com/hanb16/MKLTwoStageRO.jl) and the following [paper](https://doi.org/10.1016/j.ejor.2020.11.027):
``` bibtex
@article{han2021multiple,
  title={Multiple kernel learning-aided robust optimization: Learning algorithm, computational tractability, and usage in multi-stage decision-making},
  author={Han, Biao and Shang, Chao and Huang, Dexian},
  journal={European Journal of Operational Research},
  volume={292},
  number={3},
  pages={1004--1018},
  year={2021},
  publisher={Elsevier}
}
```

## Acknowledgments
By default, this package implicitly uses [`KernelFunctions.jl`](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl.git), open source solvers [`HiGHS.jl`](https://github.com/jump-dev/HiGHS.jl.git) and [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl.git), and the single kernel SVM solver [`LIBSVM.jl`](https://github.com/JuliaML/LIBSVM.jl.git). Thanks for these useful packages, although the user is also allowed to replace them with other alternatives. In addition, many seniors in the Julia community gave many inspiring instructions, and the author would like to thank them.


## References
1. <span id="R1">Han, B., Shang, C., & Huang, D. (2021). Multiple kernel learning-aided robust optimization: Learning algorithm, computational tractability, and usage in multi-stage decision-making. European Journal of Operational Research, 292(3), 1004-1018.
2. <span id="R2">Han, B. (2024). Multiple kernel learning-aided column-and-constraint generation method. 
3. <span id="R3">Bertsimas, D., & Shtern, S. (2018). A scalable algorithm for two-stage adaptive linear optimization. arXiv preprint arXiv:1807.02812.
4. <span id="R4">Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.
5. <span id="R5">Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.
