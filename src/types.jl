
############################################################################
#                              Basic Domains                               #
############################################################################


# --- Currently Supported Basic Domains, 
# i.e., the Available Sub-Spaces that Construct Sy and Sx ---
abstract type BasicDomain end
@kwdef struct Rplus <: BasicDomain n::Integer = 1 end # R+^n
@kwdef struct ZeroOne <: BasicDomain n::Integer = 1 end # {0, 1}^n
@kwdef struct Zplus <: BasicDomain n::Integer =1 end # Z+^n
# -----------------------------------------------------------

add_var(model, s::ZeroOne) = @variable(model, [1:s.n]; binary = true)
add_var(model, s::Rplus) = @variable(model, [1:s.n]; lower_bound = 0)
add_var(model, s::Zplus) = @variable(model, [1:s.n]; integer = true, lower_bound = 0)

Polyhedra.fulldim(S::Vector{<:BasicDomain}) = sum([s.n for s in S])

"Stretch the domain `S` into an equivalent one with dimension `fulldim(S)`. "
stretch_domain(S::Vector{<:BasicDomain}) = vcat([fill(typeof(s)(), s.n) for s in S]...)



############################################################################
#                                  Model                                   #
############################################################################

"""
    TSROModel(fields...)

Create an instance of the following two-stage robust optimization model (see `(1)` in `Ref.1` for details): 
``` formulation
Min_y c' * y + Max_{u ∈ U} Min_{x ∈ F(y, u)} b' * x

Subject to:

    A * y ≥ d, y ∈ Sy

    F(y, u) = {x ∈ Sx: G * x ≥ h - E * y - M * u}

    U is the uncertainty set
```
where `Sy` and `Sx` are allowed to be nonnegative real-valued spaces or nonnegative mixed integer spaces.

P.S. We hope the following mild assumptions hold (`Ref.1-2`):
- The Relatively Complete Recourse assumption when `Sx` is a nonnegative real-valued space, i.e., the second-stage decision problem (which is a linear program) is feasible for any given `y` and `u`.
- When `Sx` is a nonnegative mixed integer space, it has at least one real-valued dimension, i.e., the mixed integer programming recourse problem has at least one continuous recourse variable.
- The Extended Relatively Complete Recourse property when `Sx` is a nonnegative mixed integer space, i.e., the problem obtained by fixing `y`, `u` and the integer part of `x` (which is a linear program) is always feasible and bounded.
- When `Sx` is a nonnegative mixed integer space, the feasible set of discrete recouse variables (i.e., the integer part of `x`) is bounded.

# Fields
- `c::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `c` should be converted into a vector form first, e.g., `c = [2]`.
- `b::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `b` should be converted into a vector form first, e.g., `b = [1]`.
- `A::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `A` should be converted into a matrix form first, e.g., `A = [1 2]`, `A = [1; 2;;]` or `A = [0;;]`.
- `d::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `d` should be converted into a vector form first, e.g., `d = [0]`.
- `G::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `G` should be converted into a matrix form first, e.g., `G = [1 2]`, `G = [1; 2;;]` or `G = [1;;]`.
- `h::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `h` should be converted into a vector form first, e.g., `h = [0]`.
- `E::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `E` should be converted into a matrix form first, e.g., `E = [1 2]`, `E = [1; 2;;]` or `E = [-1;;]`.
- `M::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `M` should be converted into a matrix form first, e.g., `M = [1 2]`, `M = [1; 2;;]` or `M = [-1;;]`.
- `Sy::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space. It should be constructed as a vector consisting of (a mixture of) instances of the following three basic data types: `Rplus`, `ZeroOne` and `Zplus`. E.g., `Sy = [ZeroOne()]` denotes an one-dimensional binary space `{0, 1}` and `Sy = [Rplus(2), Zplus(3)]` denotes a five-dimensianal nonnegative mixed integer space `ℝ₊² × ℤ₊³`. Note that the latter is equivalent to `Sy = [fill(Rplus(), 2), fill(Zplus(), 3)]`.
- `Sx::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space. It should be constructed as a vector consisting of (a mixture of) instances of the following three basic data types: `Rplus`, `ZeroOne` and `Zplus`. The construction method is the same as `Sy`.
- `U::Union{Model, Polyhedron, StandardModel, CombinedModel, Vector{StandardModel}}`: The uncertainty set is allowed to be constructed as a `Model` using package `JuMP.jl`, a `Polyhedron` using package `Polyhedra.jl`, or a `StandardModel`, a `CombinedModel` or a `Vector{StandardModel}` using package [`MKLOneClassSVM.jl`](https://github.com/hanb16/MKLOneClassSVM.jl). This package has integrated some neccessary functions from `JuMP.jl`, `Polyhedra.jl`, `MKLOneClassSVM.jl` and some other related packages, which allows the user to model the uncertainty set conveniently without having to additionally `using/import` the above packages. Please see the `Examples` section for brief usage. P.S.: 1. Only the variables whose name is registered as `u` in a `JuMP` model uncertainty set will be recognized as the uncertainty variables unless there is only one registered variable name (or all variables are anonymous). 2. If the uncertianty set is a MKL-based one, the user should specify an `MKLCCG` algorithm to solve it.

# References
1. Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.
2. Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.
3. Bertsimas, D., & Shtern, S. (2018). A scalable algorithm for two-stage adaptive linear optimization. arXiv preprint arXiv:1807.02812.
4. Han, B. (2024). Multiple kernel learning-aided column-and-constraint generation method. 
5. Han, B., Shang, C., & Huang, D. (2021). Multiple kernel learning-aided robust optimization: Learning algorithm, computational tractability, and usage in multi-stage decision-making. European Journal of Operational Research, 292(3), 1004-1018.
"""
@kwdef struct TSROModel
    c::AbstractVector{<:Real}
    b::AbstractVector{<:Real}
    A::AbstractMatrix{<:Real}
    d::AbstractVector{<:Real}
    G::AbstractMatrix{<:Real}
    h::AbstractVector{<:Real}
    E::AbstractMatrix{<:Real}
    M::AbstractMatrix{<:Real}
    Sy::Vector{<:BasicDomain}
    Sx::Union{Vector{Rplus}, Vector{BasicDomain}}
    U::Union{Model, Polyhedron, StandardModel, CombinedModel, Vector{StandardModel}}

    function TSROModel( # Dimension Check
        c::AbstractVector{<:Real},
        b::AbstractVector{<:Real},
        A::AbstractMatrix{<:Real},
        d::AbstractVector{<:Real},
        G::AbstractMatrix{<:Real},
        h::AbstractVector{<:Real},
        E::AbstractMatrix{<:Real},
        M::AbstractMatrix{<:Real},
        Sy::Vector{<:BasicDomain},
        Sx::Union{Vector{Rplus}, Vector{BasicDomain}},
        U::Union{Model, Polyhedron, StandardModel, CombinedModel, Vector{StandardModel}}
    )
        if U isa Model
            r = length(get_uncertainty_variable(U))
        elseif U isa Polyhedron
            r = fulldim(U)
        elseif U isa MKLOCSVM_Model
            r = size(U.data, 1)
        elseif U isa Vector{StandardModel}
            r = size(U[1].data, 1)
        end
        n = fulldim(Sy); m = fulldim(Sx); 
        flag = length(c) == n && length(b) == m && (size(A, 1), size(A, 2)) == (length(d), n) && (size(G, 1), size(G, 2)) == (length(h), m) && (size(E, 1),size(E, 2)) == (length(h), n) && size(M, 1) == length(h) && size(M, 2) == r
        if ~flag
            error("Dimensions of input arguments don't match! ")
        end
        new(c, b, A, d, G, h, E, M, Sy, Sx, U)
    end
end


############################################################################
#                                Algorithms                                #
############################################################################
abstract type TSRO_Algorithm end


"""
    CCG(Fields...)

Create an instance of the (nested) column-and-constraint generation (C&CG) algorithm(`Ref.1-2`). If `Sx` is a nonnegative real-valued space, the algorithm (C&CG) from `Ref.1` will be called; if `Sx` is a nonnegative mixed integer space, the algorithm (Nested C&CG) from `Ref.2` will be called.

# Fields
- `MPsolver::Module = HiGHS`: The solver for the master problem (`MP`). It should be chosen based on the tpyes of `Sy` and `Sx`. Generally, it should be an solver that supports mixed integer linear program (MILP) if `Sy` or `Sx` is a mixed integer space.
- `SP1solver::Module = HiGHS`: The solver for the sub-problem in the KKT condition based reformulation (Bi/Tri-Equivalent I) form (`SP1`). Since `SP1` is a linearized reformulation based on big-M method, its solver should at least be an MILP solver, if `U` is just a polyhedral uncertainty set.
- `SP2solver::Module = Ipopt`: This keyword argument is only available when `Sx` is a nonnegative real-valued space. It's the solver for the sub-problem in the strong duality based reformulation form (`SP2`). The function will always try to solve `SP1` first, and then move on to `SP2` if it fails. Since `SP2` is at least a bilinear optimization problem when `U` is a polyhedron, a solver that is tailored to this or roughly one that supports non-convex quadratic program (QP) is neccessary. 
- `SP2solver_max_iter::Integer = 10000`: This keyword argument is only available when `Sx` is a nonnegative real-valued space. It's the maximum limitation of iterations for `SP2solver` to solve `SP2`.
- `SP2_MPsolver::Module = Ipopt`: This keyword argument is only available when `Sx` is a nonnegative mixed integer space. It's the solver for the master problem of the sub-problem in the strong duality based reformulation (Bi/Tri-Equivalent II) form (`SP2`). The function will always try to solve `SP1` first, and then move on to `SP2` if it fails. Since the master problem of `SP2` contains quadratic constraints even when `U` is just a polyhedral uncertainty set, an efficient solver that supports quadratically-constrained quadratic program (QCQP) is neccessary.
- `SP2_MPsolver_max_iter::Integer = 10000`: This keyword argument is only available when `Sx` is a nonnegative mixed integer space. It's the maximum limitation of iterations for `SP2_MPsolver` to solve the master problem of `SP2`.
- `SP2_SPsolver::Module = HiGHS`: This keyword argument is only available when `Sx` is a nonnegative mixed integer space.  It's the solver for the sub-problem of the sub-problem in the strong duality based reformulation (Bi/Tri-Equivalent II) form (`SP2`). The function will always try to solve `SP1` first, and then move on to `SP2` if it fails. The solver for the sub-problem of `SP2` only depends on the tpye of `Sx`, so it should be an MILP solver beacause of the mixed integer feature of `Sx`.
- `ϵ::Real = 1e-5`: The overall absolute stopping criteria of the (nested) C&CG method. If the internal quadratic programming solver is active, it's also the tolorence of it.
- `BigM::Real = 1e5`: The big-M value of `SP1` in its big-M method based linearized reformulation. Note that if a tight bound on big-M can be analytically obtained, a better performance of the algorithm can be achieved.
- `verbose::Bool = true`: The switch that controls the output of the solution process details.

# References
1. Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.
2. Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.
"""
@kwdef struct CCG <: TSRO_Algorithm
    MPsolver::Module = HiGHS
    SP1solver::Module = HiGHS
    SP2solver::Module = Ipopt
    SP2solver_max_iter::Integer = 10000
    SP2_MPsolver::Module = Ipopt
    SP2_MPsolver_max_iter::Integer = 10000
    SP2_SPsolver::Module = HiGHS
    ϵ::Real = 1e-5
    BigM::Real = 1e5
    verbose::Bool = true
end
function (ccg::CCG)(model::TSROModel)
    if model.Sx isa Vector{Rplus}
        SolveTwoStageRO_CCG(
            model.c, 
            model.b, 
            model.A, 
            model.d, 
            model.G, 
            model.h, 
            model.E, 
            model.M, 
            model.Sy, 
            model.Sx, 
            model.U;
            MPsolver = ccg.MPsolver,
            SP1solver = ccg.SP1solver,
            SP2solver = ccg.SP2solver,
            SP2solver_max_iter = ccg.SP2solver_max_iter,
            ϵ = ccg.ϵ,
            BigM = ccg.BigM,
            verbose = ccg.verbose
        )
    elseif model.Sx isa Vector{BasicDomain}
        SolveTwoStageRO_CCG(
            model.c, 
            model.b, 
            model.A, 
            model.d, 
            model.G, 
            model.h, 
            model.E, 
            model.M, 
            model.Sy, 
            model.Sx, 
            model.U;
            MPsolver = ccg.MPsolver,
            SP1solver = ccg.SP1solver,
            SP2_MPsolver = ccg.SP2_MPsolver,
            SP2_MPsolver_max_iter = ccg.SP2_MPsolver_max_iter,
            SP2_SPsolver = ccg.SP2_SPsolver,
            ϵ = ccg.ϵ,
            BigM = ccg.BigM,
            verbose = ccg.verbose
        )
    end
end




"""
    BDCP(Fields...)

Create an instance of the Benders-dual cutting plane (BDCP) algorithm (`Ref.1`).

# Fields
- `MPsolver::Module = HiGHS`: The solver for the master problem (`MP`). It should be chosen based on the tpye of `Sy`. Generally, it should be an solver that supports mixed integer linear program (MILP) if `Sy` is a mixed integer space. 
- `SP1solver::Module = HiGHS`: The solver for the sub-problem in the KKT condition based reformulation (Bi/Tri-Equivalent I) form (`SP1`). Since `SP1` is a linearized reformulation based on big-M method, its solver should at least be an MILP solver, if `U` is just a polyhedral uncertainty set.
- `SP2solver::Module = Ipopt`: The solver for the sub-problem in the strong duality based reformulation form (`SP2`). The function will always try to solve `SP1` first, and then move on to `SP2` if it fails. Since `SP2` is at least a bilinear optimization problem when `U` is a polyhedron, a solver that is tailored to this or roughly one that supports non-convex quadratic program (QP) is neccessary. 
- `SP2solver_max_iter::Integer = 10000`: The maximum limitation of iterations for `SP2solver` to solve `SP2`.
- `ϵ::Real = 1e-5`: The overall absolute stopping criteria of the (nested) C&CG method. If the internal quadratic programming solver is active, it's also the tolorence of it.
- `BigM::Real = 1e5`: The big-M value of `SP1` in its big-M method based linearized reformulation. Note that if a tight bound on big-M can be analytically obtained, a better performance of the algorithm can be achieved.
- `verbose::Bool = true`: The switch that controls the output of the solution process details.

# Reference
1. Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.
"""
@kwdef struct BDCP <: TSRO_Algorithm
    MPsolver::Module = HiGHS
    SP1solver::Module = HiGHS
    SP2solver::Module = Ipopt 
    SP2solver_max_iter::Integer = 10000
    ϵ::Real = 1e-5
    BigM::Real = 1e5
    verbose::Bool = true
end
function (bdcp::BDCP)(model::TSROModel)
    SolveTwoStageRO_BDCP(
            model.c, 
            model.b, 
            model.A, 
            model.d, 
            model.G, 
            model.h, 
            model.E, 
            model.M, 
            model.Sy, 
            model.Sx, 
            model.U;
            MPsolver = bdcp.MPsolver,
            SP1solver = bdcp.SP1solver,
            SP2solver = bdcp.SP2solver,
            SP2solver_max_iter = bdcp.SP2solver_max_iter,
            ϵ = bdcp.ϵ,
            BigM = bdcp.BigM,
            verbose = bdcp.verbose
        )
end


"""
    ECCG(Fields...)

Create an instance of the Extended Column-and-Constraint Generation (ECCG) algorithm (`Ref.1`).

# Fields
- `MPsolver::Module = HiGHS`: The solver for the master problem (`MP`). It should be chosen based on the tpyes of `Sx` and `Sy`. Generally, it should be an solver that supports mixed integer linear program (MILP) if `Sx` or `Sy` is a mixed integer space.
- `SPsolver::Module = HiGHS`: The solver for the sub-problem (`SP`) in the KKT condition based reformulation form (both the feasiblity oracle and the optimality oracle). Since the sub-problem oracles are linearized reformulations based on big-M method, their solver should at least be an MILP solver, if `U` is just a polyhedral uncertainty set.
- `ϵ::Real = 1e-5`: The overall absolute stopping criteria of the Extended CCG method. It's also used as the tolorence in the feasiblity oracle for which if the objective value of the oracle is not less than `ϵ` we then think the current `x` is infeasible.
- `BigM::Real = 1e5`: The big-M value of `SP` in its big-M method based linearized reformulation (both the feasiblity oracle and the optimality oracle). Note that if a tight bound on big-M can be analytically obtained, a better performance of the algorithm can be achieved.
- `verbose::Bool = true`: The switch that controls the output of the solution process details.

# Reference
1. Bertsimas, D., & Shtern, S. (2018). A scalable algorithm for two-stage adaptive linear optimization. arXiv preprint arXiv:1807.02812.
"""
@kwdef struct ECCG <: TSRO_Algorithm
    MPsolver::Module = HiGHS
    SPsolver::Module = HiGHS
    ϵ::Real = 1e-5
    BigM::Real = 1e5
    verbose::Bool = true
end
function (eccg::ECCG)(model::TSROModel)
    SolveTwoStageRO_ECCG(
            model.c, 
            model.b, 
            model.A, 
            model.d, 
            model.G, 
            model.h, 
            model.E, 
            model.M, 
            model.Sy, 
            model.Sx, 
            model.U;
            MPsolver = eccg.MPsolver,
            SPsolver = eccg.SPsolver,
            ϵ = eccg.ϵ,
            BigM = eccg.BigM, 
            verbose = eccg.verbose
        )
end



"""
    MKLCCG(Fields...)

Create an instance of the Multiple Kernel Learning-aided Column-and-Constraint Generation (MKLCCG) algorithm (`Ref.1`).

# Fields
- `MPsolver::Module = HiGHS`: The solver for the master problem (`MP`). It should be chosen based on the tpyes of `Sx` and `Sy`. Generally, it should be an solver that supports mixed integer linear program (MILP) if `Sx` or `Sy` is a mixed integer space.
- `SPsolver::Module = HiGHS`: The solver for the sub-problem (`SP`) in the KKT condition based reformulation form (both the feasiblity oracle and the optimality oracle). Since the sub-problem oracles are linearized reformulations based on big-M method, their solver should at least be an MILP solver, if `U` is just a polyhedral uncertainty set.
- `ϵ::Real = 1e-5`: The overall absolute stopping criteria of the Extended CCG method. It's also used as the tolorence in the feasiblity oracle for which if the objective value of the oracle is not less than `ϵ` we then think the current `x` is infeasible.
- `BigM::Real = 1e5`: The big-M value of `SP` in its big-M method based linearized reformulation (both the feasiblity oracle and the optimality oracle). Note that if a tight bound on big-M can be analytically obtained, a better performance of the algorithm can be achieved.
- `verbose::Bool = true`: The switch that controls the output of the solution process details.


# Reference
1. Han, B. (2024). Multiple kernel learning-aided column-and-constraint generation method. 
2. Han, B., Shang, C., & Huang, D. (2021). Multiple kernel learning-aided robust optimization: Learning algorithm, computational tractability, and usage in multi-stage decision-making. European Journal of Operational Research, 292(3), 1004-1018.

"""
@kwdef struct MKLCCG <: TSRO_Algorithm
    MPsolver::Module = HiGHS
    SPsolver::Module = HiGHS
    ϵ::Real = 1e-5
    BigM::Real = 1e5
    verbose::Bool = true
end
function (mklccg::MKLCCG)(model::TSROModel)
    SolveTwoStageRO_MKLCCG(
        model.c, 
        model.b, 
        model.A, 
        model.d, 
        model.G, 
        model.h, 
        model.E, 
        model.M, 
        model.Sy, 
        model.Sx, 
        model.U;
        MPsolver = mklccg.MPsolver,
        SPsolver = mklccg.SPsolver,
        ϵ = mklccg.ϵ,
        BigM = mklccg.BigM, 
        verbose = mklccg.verbose
    )
end



