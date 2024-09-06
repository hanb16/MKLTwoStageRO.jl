
"""
    SolveTwoStageRO_CCG(args...; kwargs...)

Solve the following two-stage robust optimization problems using (nested) column-and-constraint generation (C&CG) method (`Ref.1-2`):
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

# Arguments
- `c::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `c` should be converted into a vector form first, e.g., `c = [2]`.
- `b::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `b` should be converted into a vector form first, e.g., `b = [1]`.
- `A::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `A` should be converted into a matrix form first, e.g., `A = [1 2]`, `A = [1; 2;;]` or `A = [0;;]`.
- `d::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `d` should be converted into a vector form first, e.g., `d = [0]`.
- `G::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `G` should be converted into a matrix form first, e.g., `G = [1 2]`, `G = [1; 2;;]` or `G = [1;;]`.
- `h::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `h` should be converted into a vector form first, e.g., `h = [0]`.
- `E::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `E` should be converted into a matrix form first, e.g., `E = [1 2]`, `E = [1; 2;;]` or `E = [-1;;]`.
- `M::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `M` should be converted into a matrix form first, e.g., `M = [1 2]`, `M = [1; 2;;]` or `M = [-1;;]`.
- `Sy::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space. It should be constructed as a vector consisting of (a mixture of) instances of the following three basic data types: `Rplus`, `ZeroOne` and `Zplus`. E.g., `Sy = [ZeroOne()]` denotes an one-dimensional binary space `{0, 1}` and `Sy = [Rplus(2), Zplus(3)]` denotes a five-dimensianal nonnegative mixed integer space `ℝ₊² × ℤ₊³`. Note that the latter is equivalent to `Sy = [fill(Rplus(), 2), fill(Zplus(), 3)]`.
- `Sx::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space. If it is the former, the algorithm (C&CG) from `Ref.1` will be called; if it is the latter, the algorithm (Nested C&CG) from `Ref.2` will be called. It should be constructed as a vector consisting of (a mixture of) instances of the following three basic data types: `Rplus`, `ZeroOne` and `Zplus`. The construction method is the same as `Sy`.
- `U::Union{Model, Polyhedron}`: The uncertainty set is allowed to be constructed as a `Model` using package `JuMP.jl` or a `Polyhedron` using package `Polyhedra.jl`. This package has integrated some neccessary functions from `JuMP.jl`, `Polyhedra.jl` and some other related packages, which allows the user to model the uncertainty set conveniently without having to additionally `using/import` the above packages. Please see the `Examples` section for brief usage. Note that only the variables whose name is registered as `u` in a `JuMP` model uncertainty set will be recognized as the uncertainty variables unless there is only one registered variable name (or all variables are anonymous).

# Keywords
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

# Returns
This function will return a `Tuple{Vector{Real}, Real, Integer}` whose entries are in order:
- `y::Vector{Real}`: the optimal solution to the first stage decision variable `y`.
- `objv::Real`: the optimal objective value of the two-stage robust optimization problem.
- `k::Integer`: the total number of iterations when the solution process terminates.

# Examples
``` julia
using MKLTwoStageRO
```

### Case-1:
``` julia
c = [2]; b = [1];
A = [0;;]; d = [0]; 
G = [1;;]; h = [0]; E = [-1;;]; M = [-1;;];
Sy = [ZeroOne()]; Sx = [Rplus()]; 
# Call the function with a JuMP `Model` type uncertianty set:
UncMod = Model()
@variable(UncMod, 0 <= δ <= 1)
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod) 
# Call the function with a `Polyhedron` type uncertianty set:
UncSet = polyhedron(HalfSpace([1], 1) ∩ HalfSpace([-1], 0))
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
```

### Case-2:
``` julia
c = [1, 2]; b = [3];
A = [4 5]; d = [6]; 
G = [1;;]; h = [2]; E = [-3 -4]; M = [-5 -6];
Sx = [Rplus()]; Sy = [Rplus(), ZeroOne()];
# Call the function with a JuMP `Model` type uncertianty set:
UncMod = Model()
u0 = [0, 0]
@variable(UncMod, u[1:2])
@variable(UncMod, -1 <= δ[1:2] <= 1)
@constraint(UncMod, u == u0 + δ)
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
# Call the function with a `Polyhedron` type uncertianty set:
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))
ind_δ = ind_of_var(UncMod, "δ")
UncSet = eliminate(UncSet, ind_δ)
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
```

### Case-3:
``` julia
c = [1]; b = [2; 3];
A = [4;;]; d = [5];
G = [6 7]; h = [8]; E = [10;;]; M = [9;;];
Sy = [Rplus()]; Sx = [Rplus(); ZeroOne()];
# Call the function with a JuMP `Model` type uncertianty set:
UncMod = Model()
@variable(UncMod, u)
δ = @expression(UncMod, 2u - 1)
@constraint(UncMod, -1 <= δ <= 1)
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
# Call the function with a `Polyhedron` type uncertianty set:
UncSet = polyhedron(convexhull([0], [1]))
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
```


# References
1. Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.
2. Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.
"""
function SolveTwoStageRO_CCG end

############################################################################
#                        For General Uncertainty Set                       #
############################################################################

function SolveTwoStageRO_CCG(
    c::AbstractVector{<:Real}, 
    b::AbstractVector{<:Real}, 
    A::AbstractMatrix{<:Real}, 
    d::AbstractVector{<:Real}, 
    G::AbstractMatrix{<:Real}, 
    h::AbstractVector{<:Real}, 
    E::AbstractMatrix{<:Real}, 
    M::AbstractMatrix{<:Real}, 
    Sy::Vector{<:BasicDomain}, 
    Sx::Vector{Rplus}, 
    U::Model; 
    MPsolver::Module = HiGHS, 
    SP1solver::Module = HiGHS, 
    SP2solver::Module = Ipopt, 
    SP2solver_max_iter::Integer = 10000, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true
)

    n = fulldim(Sy); m = fulldim(Sx); r = length(get_uncertainty_variable(U))

    k = 0
    LB = -Inf; UB = +Inf

    # --- Master Problem Initialization ---
    MP = Model(MPsolver.Optimizer)
    set_silent(MP)
    y = reduce(vcat, add_var.(MP, Sy))
    @variable(MP, η)
    @constraint(MP, A * y >= d)
    # -------------------------------------

    @objective(MP, Min, c' * y)
    optimize!(MP) 
    y_star = value.(y); η_star = -Inf
    LB = c' * y_star + η_star
    @objective(MP, Min, c' * y + η)

    if verbose
        @printf("|----- Column & Constraint Generation Method -----|\n")
        @printf("|   Iter.   |    L.B.    |    U.B.    |    Gap    |\n")
        @printf("|-----------|------------|------------|-----------|\n")
        LBstr = num2str(LB, 2, 9)
        UBstr = num2str(UB, 2, 9)
        GAPstr = num2str(UB - LB, 2, 8) 
        @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
    end
    
    # --- Sub-Problem Initialization ---
    # KKT condition based reformulation (SP1):
    SP1 = copy(U)
    set_optimizer(SP1, SP1solver.Optimizer)
    set_silent(SP1)
    u1 = get_uncertainty_variable(SP1)
    x = reduce(vcat, add_var.(SP1, Sx))
    π1 = @variable(SP1, [eachindex(h)]; lower_bound = 0)
    w = @variable(SP1, [eachindex(h)]; binary = true)
    v = @variable(SP1, [1:m]; binary = true)
    @constraint(SP1, con1, G * x >= h - E * y_star - M * u1)
    @constraint(SP1, G' * π1 <= b)
    @constraint(SP1, π1 <= BigM * w)
    @constraint(SP1, con2, G * x - h + E * y_star + M * u1 <= BigM * (1 .- w))
    @constraint(SP1, x <= BigM * v)
    @constraint(SP1, b - G' * π1 <= BigM * (1 .- v))
    @objective(SP1, Max, b' * x)

    # Strong duality based reformulation (SP2):
    SP2 = copy(U)
    set_optimizer(SP2, SP2solver.Optimizer)
    set_optimizer_attributes(SP2, "tol" => ϵ, "max_iter" => SP2solver_max_iter)
    set_silent(SP2)
    u2 = get_uncertainty_variable(SP2)
    π2 = @variable(SP2, [eachindex(h)]; lower_bound = 0)
    @constraint(SP2, G' * π2 <= b)
    @objective(SP2, Max, (h - E * y_star - M * u2)' * π2)
    # ----------------------------------

    while true
        optimize!(SP1)
        if termination_status(SP1) == OPTIMAL
            Q_y_star = objective_value(SP1)
            u_star = value.(u1)
        else
            Q_y_star = +Inf
            if k == 0
                # --- To get the first u_star ---
                UncMod = copy(U)
                set_optimizer(UncMod, SP1solver.Optimizer)
                set_silent(UncMod)
                u0 = get_uncertainty_variable(UncMod)
                @objective(UncMod, Max, sum(u0))
                optimize!(UncMod)
                u_star = value.(u0)
                # -------------------------------
            else
                optimize!(SP2)
                u_star = value.(u2)
            end
        end
        UB = min(UB, c' * y_star + Q_y_star)

        if UB - LB <= ϵ
            if verbose
                k += 1
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    |    ----    | %9s  | %8s  |\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MP ---
        x_new = reduce(vcat, add_var.(MP, Sx))
        @constraint(MP, η >= b' * x_new)
        @constraint(MP, E * y + G * x_new >= h - M * u_star)
        # -----------------------------

        optimize!(MP)
        y_star = value.(y); η_star = value.(η)
        LB = c' * y_star + η_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

        # --- Update Sub-Problem ---
        # KKT condition based reformulation (SP1):
        delete(SP1, con1); unregister(SP1, :con1)
        delete(SP1, con2); unregister(SP1, :con2)
        @constraint(SP1, con1, G * x >= h - E * y_star - M * u1)
        @constraint(SP1, con2, G * x - h + E * y_star + M * u1 <= BigM * (1 .- w))

        # Strong duality based reformulation (SP2):
        @objective(SP2, Max, (h - E * y_star - M * u2)' * π2)
        # --------------------------

    end

    objv = UB

    return value.(y), objv, k
end

function SolveTwoStageRO_CCG(
    c::AbstractVector{<:Real}, 
    b::AbstractVector{<:Real}, 
    A::AbstractMatrix{<:Real}, 
    d::AbstractVector{<:Real}, 
    G::AbstractMatrix{<:Real}, 
    h::AbstractVector{<:Real}, 
    E::AbstractMatrix{<:Real}, 
    M::AbstractMatrix{<:Real}, 
    Sy::Vector{<:BasicDomain}, 
    Sx::Vector{BasicDomain}, 
    U::Model; 
    MPsolver::Module = HiGHS, 
    SP1solver::Module = HiGHS, 
    SP2_MPsolver::Module = Ipopt, 
    SP2_MPsolver_max_iter::Integer = 10000, 
    SP2_SPsolver::Module = HiGHS, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true,
)

    n = fulldim(Sy); m = fulldim(Sx); r = length(get_uncertainty_variable(U))

    # --- Problem Reformulation ---
    c̄, b̄, Ā, d̄, Ḡ, h̄, Ē, M̄, S̄y, S̄x = c, b, A, d, G, h, E, M, Sy, Sx

    S̄y = stretch_domain(S̄y)
    S̄x = stretch_domain(S̄x)

    Sy = S̄y
    ind_Rplus = typeof.(S̄x) .== Rplus
    Sx = S̄x[ind_Rplus]
    Sz = S̄x[.~ind_Rplus]

    c = c̄'
    d = b̄[ind_Rplus]'
    g = b̄[.~ind_Rplus]'
    A = Ā
    b = d̄
    E = Ḡ[:, ind_Rplus]
    G = Ḡ[:, .~ind_Rplus]
    f = h̄
    D = Ē
    R = M̄

    m = fulldim(Sy)
    p = fulldim(Sx)
    n = fulldim(Sz)
    q = r
    # -----------------------------

    k = 0
    LB = -Inf; UB = +Inf

    # --- Master Problem Initialization ---
    MP = Model(MPsolver.Optimizer)
    set_silent(MP)
    y = reduce(vcat, add_var.(MP, Sy))
    @variable(MP, η)
    @constraint(MP, A * y >= b)
    # -------------------------------------

    @objective(MP, Min, c * y)
    optimize!(MP) 
    y_star = value.(y); η_star = -Inf
    LB = c * y_star + η_star
    @objective(MP, Min, c * y + η)

    if verbose
        @printf("|----- Column & Constraint Generation Method -----|\n")
        @printf("|   Iter.   |    L.B.    |    U.B.    |    Gap    |\n")
        @printf("|-----------|------------|------------|-----------|\n")
        LBstr = num2str(LB, 2, 9)
        UBstr = num2str(UB, 2, 9)
        GAPstr = num2str(UB - LB, 2, 8) 
        @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
    end

    # --- Sub-Problem Initialization: to get the first u_star ---
    UncMod = copy(U)
    set_optimizer(UncMod, SP1solver.Optimizer)
    set_silent(UncMod)
    u0 = get_uncertainty_variable(UncMod)
    @objective(UncMod, Max, sum(u0))
    optimize!(UncMod)
    u_star = value.(u0)
    # -----------------------------------------------------------

    while true
        # --- Call the Oracle to Solve Sub-Problem ---
        u_star, Q_star = SP_Oracle(
            y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U; 
            SP1solver = SP1solver, 
            SP2_MPsolver = SP2_MPsolver, 
            SP2_MPsolver_max_iter = SP2_MPsolver_max_iter,
            SP2_SPsolver = SP2_SPsolver, 
            ϵ = ϵ, 
            BigM = BigM, 
            verbose = verbose,
        )

        UB = min(UB, c * y_star + Q_star)
        # --------------------------------------------

        if UB - LB <= ϵ
            if verbose
                k += 1
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    |    ----    | %9s  | %8s  |\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MP ---
        x_new = reduce(vcat, add_var.(MP, Sx))
        z_new = reduce(vcat, add_var.(MP, Sz))
        @constraint(MP, η >= d * x_new + g * z_new)
        @constraint(MP, D * y + E * x_new + G * z_new >= f - R * u_star)
        # -----------------------------

        optimize!(MP)
        y_star = value.(y); η_star = value.(η)
        LB = c * y_star + η_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

    end

    objv = UB

    return value.(y), objv, k
end




"""
    InnerLevel_CCG_EquivalentI(args..., U::Model; kwargs...)

Re-apply the CC&G strategy to solve the SP of KKT condition based reformulation (Bi/Tri-Equivalent I).
"""
function InnerLevel_CCG_EquivalentI(
    y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U::Model; 
    solver = HiGHS, 
    ϵ, 
    BigM, 
    verbose
)

    r = length(get_uncertainty_variable(U))
    Sπ = [Rplus(length(f))]
    Sδ = [ZeroOne(length(d))] # For the auxiliary variable of x_new
    Sγ = [ZeroOne(length(f))] # For the auxiliary variable of π_new

    k = 0
    LB = -Inf; UB = +Inf
    term_stat = ""

    # --- Master Problem (MPs) Initialization ---
    MPs = copy(U)
    set_optimizer(MPs, solver.Optimizer)
    set_silent(MPs)
    u = get_uncertainty_variable(MPs)
    θ = @variable(MPs)
    @objective(MPs, Max, θ)
    # -------------------------------------------

    Q_star = +Inf
    UB = Q_star
    
    # --- Sub-Problem (SPs) Initialization ---
    SPs = Model(solver.Optimizer)
    set_silent(SPs)
    x = reduce(vcat, add_var.(SPs, Sx))
    z = reduce(vcat, add_var.(SPs, Sz))
    @objective(SPs, Min, d * x + g * z)
    @constraint(SPs, con, E * x + G * z >= f - R * u_star - D * y_star)
    # ----------------------------------------

    while true
        optimize!(SPs)
        if termination_status(SPs) == OPTIMAL
            SPs_star = objective_value(SPs)
            z_star = value.(z)
        else
            if verbose
                k += 1
                term_stat = "SPs " * string(termination_status(SPs))
                @printf("┆Eqv.I ┆%3s ┆ Fail to calc. L.B.: %14s. ┆\n", k, term_stat)
            end
            break
        end
        LB = max(LB, SPs_star)

        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            if verbose
                k += 1
                LBstr = num2str(LB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("┆Eqv.I ┆%3s ┆ %9s  ┆    ----    ┆ %8s  ┆\n", k, LBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MPs ---
        x_new = reduce(vcat, add_var.(MPs, Sx))
        π_new = reduce(vcat, add_var.(MPs, Sπ))
        δ_new = reduce(vcat, add_var.(MPs, Sδ))
        γ_new = reduce(vcat, add_var.(MPs, Sγ))
        @constraint(MPs, θ <= g * z_star + d * x_new)
        @constraint(MPs, E * x_new >= f - R * u - D * y_star - G * z_star)
        @constraint(MPs, E' * π_new <= d')
        @constraint(MPs, x_new <= BigM * δ_new)
        @constraint(MPs, d' - E' * π_new <= BigM * (1 .- δ_new))
        @constraint(MPs, π_new <= BigM * γ_new)
        @constraint(MPs, E * x_new - f + R * u + D * y_star + G * z_star <= BigM * (1 .- γ_new))
        # ------------------------------

        optimize!(MPs)
        if termination_status(SPs) == OPTIMAL
            Q_star = objective_value(MPs)
            u_star = value.(u)
        else
            if verbose
                k += 1
                term_stat = "MPs " * string(termination_status(MPs))
                @printf("┆Eqv.I ┆%3s ┆ Fail to calc. U.B.: %14s. ┆\n", k, term_stat)
            end
            break
        end
        UB = Q_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("┆Eqv.I ┆%3s ┆ %9s  ┆ %9s  ┆ %8s  ┆\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            break
        end

        # --- Update Sub-Problem (SPs) ---
        # Alternative method:
        delete(SPs, con); unregister(SPs, :con)
        @constraint(SPs, con, E * x + G * z >= f - R * u_star - D * y_star)
        # --------------------------------

    end

    return u_star, Q_star, term_stat

end




"""
    InnerLevel_CCG_EquivalentII(args..., U::Model; kwargs...)

If the extended relately complete recourse assumption does not hold, the KKT condition based reformulation (Bi/Tri-Equivalent I) is not valid. Now, the strong duality based reformulation (Bi/Tri-Equivalent II) is still valid and the corresponding C&CG variant is applicable. So here we take it as a standby method.
"""
function InnerLevel_CCG_EquivalentII(
    y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U::Model; 
    MPsolver = Ipopt, 
    MPsolver_max_iter,
    SPsolver = HiGHS, 
    ϵ, 
    verbose
)

    r = length(get_uncertainty_variable(U))
    Sπ = [Rplus(length(f))]

    k = 0
    LB = -Inf; UB = +Inf
    term_stat = ""

    # --- Master Problem (MPs) Initialization ---
    MPs = copy(U)
    set_optimizer_attributes(MPs, "tol" => ϵ, "max_iter" => MPsolver_max_iter)
    set_silent(MPs)
    u = get_uncertainty_variable(MPs)
    θ = @variable(MPs)
    @objective(MPs, Max, θ)
    # -------------------------------------------  

    Q_star = +Inf
    UB = Q_star

    while true
        # --- Call Another C&CG Procedure to Solve SPs ---
        z_star, SPs_star = InnerLevel_CCG_EquivalentII_SSP(
            y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, Sπ; 
            solver = SPsolver, 
            ϵ = 1e-5, 
            verbose = true
        )
        # ------------------------------------------------
        LB = max(LB, g * z_star + SPs_star)

        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            if verbose
                k += 1
                LBstr = num2str(LB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("┆Eqv.II┆%3s ┆ %9s  ┆    ----    ┆ %8s  ┆\n", k, LBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MPs ---
        π_new = reduce(vcat, add_var.(MPs, Sπ))
        @constraint(MPs, θ <= g * z_star + (f - R * u - D * y_star - G * z_star)' * π_new)
        @constraint(MPs, E' * π_new <= d')
        # ------------------------------

        optimize!(MPs)
        if termination_status(MPs) == OPTIMAL
            k += 1
            u_star = value.(u)
            Q_star = objective_value(MPs)
        elseif termination_status(MPs) == LOCALLY_SOLVED
            if verbose
                term_stat = "MPs " * string(termination_status(MPs))
                k += 1
                u_star = value.(u)
                Q_star = objective_value(MPs)
                LBstr = num2str(LB, 2, 9)
                @printf("┆Eqv.II┆%3s ┆ %9s  ┆     LOCALLY_SOLVED     ┆\n", k, LBstr)
            end
            break
        else
            if verbose
                k += 1
                term_stat = "MPs " * string(termination_status(MPs))
                @printf("┆Eqv.II┆%3s ┆ Fail to calc. U.B.: %14s. ┆\n", k, term_stat)
            end
            break
        end
        UB = Q_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("┆Eqv.II┆%3s ┆ %9s  ┆ %9s  ┆ %8s  ┆\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            break
        end
    end

    return u_star, Q_star, term_stat
end






############################################################################
#                      For Polyhedral Uncertainty Set                      #
############################################################################

function SolveTwoStageRO_CCG(
    c::AbstractVector{<:Real}, 
    b::AbstractVector{<:Real}, 
    A::AbstractMatrix{<:Real}, 
    d::AbstractVector{<:Real}, 
    G::AbstractMatrix{<:Real}, 
    h::AbstractVector{<:Real}, 
    E::AbstractMatrix{<:Real}, 
    M::AbstractMatrix{<:Real}, 
    Sy::Vector{<:BasicDomain}, 
    Sx::Vector{Rplus}, 
    U::Polyhedron; 
    MPsolver::Module = HiGHS, 
    SP1solver::Module = HiGHS, 
    SP2solver::Module = Ipopt, 
    SP2solver_max_iter::Integer = 10000, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true
)

    n = fulldim(Sy); m = fulldim(Sx); r = fulldim(U)

    k = 0
    LB = -Inf; UB = +Inf

    # --- Master Problem Initialization ---
    MP = Model(MPsolver.Optimizer)
    set_silent(MP)
    y = reduce(vcat, add_var.(MP, Sy))
    @variable(MP, η)
    @constraint(MP, A * y >= d)
    # -------------------------------------

    @objective(MP, Min, c' * y)
    optimize!(MP) 
    y_star = value.(y); η_star = -Inf
    LB = c' * y_star + η_star
    @objective(MP, Min, c' * y + η)

    if verbose
        @printf("|----- Column & Constraint Generation Method -----|\n")
        @printf("|   Iter.   |    L.B.    |    U.B.    |    Gap    |\n")
        @printf("|-----------|------------|------------|-----------|\n")
        LBstr = num2str(LB, 2, 9)
        UBstr = num2str(UB, 2, 9)
        GAPstr = num2str(UB - LB, 2, 8) 
        @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
    end

    # --- Sub-Problem Initialization ---
    # KKT condition based reformulation (SP1):
    SP1 = Model(SP1solver.Optimizer)
    set_silent(SP1)
    x = reduce(vcat, add_var.(SP1, Sx))
    @variable(SP1, u1[1:r] ∈ U)
    @variable(SP1, π1[eachindex(h)] >= 0)
    @variable(SP1, w[eachindex(h)], Bin)
    @variable(SP1, v[1:m], Bin)
    @constraint(SP1, con1, G * x >= h - E * y_star - M * u1)
    @constraint(SP1, G' * π1 <= b)
    @constraint(SP1, π1 <= BigM * w)
    @constraint(SP1, con2, G * x - h + E * y_star + M * u1 <= BigM * (1 .- w))
    @constraint(SP1, x <= BigM * v)
    @constraint(SP1, b - G' * π1 <= BigM * (1 .- v))
    @objective(SP1, Max, b' * x)

    # Strong duality based reformulation (SP2):
    SP2 = Model(SP2solver.Optimizer)
    set_optimizer_attributes(SP2, "tol" => ϵ, "max_iter" => SP2solver_max_iter)
    set_silent(SP2)
    @variable(SP2, u2[1:r] ∈ U)
    @variable(SP2, π2[eachindex(h)] >= 0)
    @constraint(SP2, G' * π2 <= b)
    @objective(SP2, Max, (h - E * y_star - M * u2)' * π2)
    # ----------------------------------

    while true
        optimize!(SP1)
        if termination_status(SP1) == OPTIMAL
            Q_y_star = objective_value(SP1)
            u_star = value.(u1)
        else
            Q_y_star = +Inf
            if k == 0
                # --- To get the first u_star ---
                UncMod = Model(SP1solver.Optimizer)
                set_silent(UncMod)
                @variable(UncMod, u0[1:r] ∈ U)
                @objective(UncMod, Max, sum(u0))
                optimize!(UncMod)
                u_star = value.(u0)
                # -------------------------------
            else
                optimize!(SP2)
                u_star = value.(u2)
            end
        end
        UB = min(UB, c' * y_star + Q_y_star)

        if UB - LB <= ϵ
            if verbose
                k += 1
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    |    ----    | %9s  | %8s  |\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MP ---
        x_new = reduce(vcat, add_var.(MP, Sx))
        @constraint(MP, η >= b' * x_new)
        @constraint(MP, E * y + G * x_new >= h - M * u_star)
        # -----------------------------

        optimize!(MP)
        y_star = value.(y); η_star = value.(η)
        LB = c' * y_star + η_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

        # --- Update Sub-Problem ---
        # KKT condition based reformulation (SP1):
        delete(SP1, con1); unregister(SP1, :con1)
        delete(SP1, con2); unregister(SP1, :con2)
        @constraint(SP1, con1, G * x >= h - E * y_star - M * u1)
        @constraint(SP1, con2, G * x - h + E * y_star + M * u1 <= BigM * (1 .- w))

        # Strong duality based reformulation (SP2):
        @objective(SP2, Max, (h - E * y_star - M * u2)' * π2)
        # --------------------------

    end

    objv = UB

    return value.(y), objv, k
end

function SolveTwoStageRO_CCG(
    c::AbstractVector{<:Real}, 
    b::AbstractVector{<:Real}, 
    A::AbstractMatrix{<:Real}, 
    d::AbstractVector{<:Real}, 
    G::AbstractMatrix{<:Real}, 
    h::AbstractVector{<:Real}, 
    E::AbstractMatrix{<:Real}, 
    M::AbstractMatrix{<:Real}, 
    Sy::Vector{<:BasicDomain}, 
    Sx::Vector{BasicDomain}, 
    U::Polyhedron; 
    MPsolver::Module = HiGHS, 
    SP1solver::Module = HiGHS, 
    SP2_MPsolver::Module = Ipopt, 
    SP2_MPsolver_max_iter::Integer = 10000, 
    SP2_SPsolver::Module = HiGHS, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true,
)

    n = fulldim(Sy); m = fulldim(Sx); r = fulldim(U)

    # --- Problem Reformulation ---
    c̄, b̄, Ā, d̄, Ḡ, h̄, Ē, M̄, S̄y, S̄x = c, b, A, d, G, h, E, M, Sy, Sx

    S̄y = stretch_domain(S̄y)
    S̄x = stretch_domain(S̄x)

    Sy = S̄y
    ind_Rplus = typeof.(S̄x) .== Rplus
    Sx = S̄x[ind_Rplus]
    Sz = S̄x[.~ind_Rplus]

    c = c̄'
    d = b̄[ind_Rplus]'
    g = b̄[.~ind_Rplus]'
    A = Ā
    b = d̄
    E = Ḡ[:, ind_Rplus]
    G = Ḡ[:, .~ind_Rplus]
    f = h̄
    D = Ē
    R = M̄

    m = fulldim(Sy)
    p = fulldim(Sx)
    n = fulldim(Sz)
    q = fulldim(U)
    # -----------------------------

    k = 0
    LB = -Inf; UB = +Inf

    # --- Master Problem Initialization ---
    MP = Model(MPsolver.Optimizer)
    set_silent(MP)
    y = reduce(vcat, add_var.(MP, Sy))
    @variable(MP, η)
    @constraint(MP, A * y >= b)
    # -------------------------------------

    @objective(MP, Min, c * y)
    optimize!(MP) 
    y_star = value.(y); η_star = -Inf
    LB = c * y_star + η_star
    @objective(MP, Min, c * y + η)

    if verbose
        @printf("|----- Column & Constraint Generation Method -----|\n")
        @printf("|   Iter.   |    L.B.    |    U.B.    |    Gap    |\n")
        @printf("|-----------|------------|------------|-----------|\n")
        LBstr = num2str(LB, 2, 9)
        UBstr = num2str(UB, 2, 9)
        GAPstr = num2str(UB - LB, 2, 8) 
        @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
    end

    # --- Sub-Problem Initialization: to get the first u_star ---
    UncMod = Model(SP1solver.Optimizer)
    set_silent(UncMod)
    @variable(UncMod, u0[1:r] ∈ U)
    @objective(UncMod, Max, sum(u0))
    optimize!(UncMod)
    u_star = value.(u0)
    # -----------------------------------------------------------

    while true
        # --- Call the Oracle to Solve Sub-Problem ---
        u_star, Q_star = SP_Oracle(
            y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U; 
            SP1solver = SP1solver, 
            SP2_MPsolver = SP2_MPsolver, 
            SP2_MPsolver_max_iter = SP2_MPsolver_max_iter,
            SP2_SPsolver = SP2_SPsolver, 
            ϵ = ϵ, 
            BigM = BigM, 
            verbose = verbose,
        )

        UB = min(UB, c * y_star + Q_star)
        # --------------------------------------------

        if UB - LB <= ϵ
            if verbose
                k += 1
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    |    ----    | %9s  | %8s  |\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MP ---
        x_new = reduce(vcat, add_var.(MP, Sx))
        z_new = reduce(vcat, add_var.(MP, Sz))
        @constraint(MP, η >= d * x_new + g * z_new)
        @constraint(MP, D * y + E * x_new + G * z_new >= f - R * u_star)
        # -----------------------------

        optimize!(MP)
        y_star = value.(y); η_star = value.(η)
        LB = c * y_star + η_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

    end

    objv = UB

    return value.(y), objv, k
end




"""
    SP_Oracle(args...; kwargs...)

The oracle to solve the sub-problem.
"""
function SP_Oracle(
    y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U; 
    SP1solver = HiGHS, 
    SP2_MPsolver = Ipopt, 
    SP2_MPsolver_max_iter,
    SP2_SPsolver = HiGHS,
    ϵ, 
    BigM, 
    verbose,
)
    # KKT condition based reformulation - Bi/Tri-Equivalent I (SP1):
    u_star, Q_star, SP1_term_stat = InnerLevel_CCG_EquivalentI(
        y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U; 
        solver = SP1solver, 
        ϵ = ϵ, 
        BigM = BigM, 
        verbose = verbose
    )

    # Strong duality based reformulation - Bi/Tri-Equivalent II (SP2):
    if SP1_term_stat != "OPTIMAL"
        u_star, Q_star = InnerLevel_CCG_EquivalentII(
            y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U; 
            MPsolver = SP2_MPsolver,
            MPsolver_max_iter = SP2_MPsolver_max_iter,
            SPsolver = SP2_SPsolver,
            ϵ = ϵ, 
            verbose = verbose
        )
    end

    return u_star, Q_star
end




"""
    InnerLevel_CCG_EquivalentI(args..., U::Polyhedron; kwargs...)

Re-apply the CC&G strategy to solve the SP of KKT condition based reformulation (Bi/Tri-Equivalent I).
"""
function InnerLevel_CCG_EquivalentI(
    y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U::Polyhedron; 
    solver = HiGHS, 
    ϵ, 
    BigM, 
    verbose
)

    r = fulldim(U)
    Sπ = [Rplus(length(f))]
    Sδ = [ZeroOne(length(d))] # For the auxiliary variable of x_new
    Sγ = [ZeroOne(length(f))] # For the auxiliary variable of π_new

    k = 0
    LB = -Inf; UB = +Inf
    term_stat = ""

    # --- Master Problem (MPs) Initialization ---
    MPs = Model(solver.Optimizer)
    set_silent(MPs)
    @variable(MPs, u[1:r] ∈ U)
    @variable(MPs, θ)
    @objective(MPs, Max, θ)
    # -------------------------------------------

    Q_star = +Inf
    UB = Q_star

    # --- Sub-Problem (SPs) Initialization ---
    SPs = Model(solver.Optimizer)
    set_silent(SPs)
    x = reduce(vcat, add_var.(SPs, Sx))
    z = reduce(vcat, add_var.(SPs, Sz))
    @objective(SPs, Min, d * x + g * z)
    @constraint(SPs, con, E * x + G * z >= f - R * u_star - D * y_star)
    # ----------------------------------------

    while true
        optimize!(SPs)
        if termination_status(SPs) == OPTIMAL
            SPs_star = objective_value(SPs)
            z_star = value.(z)
        else
            if verbose
                k += 1
                term_stat = "SPs " * string(termination_status(SPs))
                @printf("┆Eqv.I ┆%3s ┆ Fail to calc. L.B.: %14s. ┆\n", k, term_stat)
            end
            break
        end
        LB = max(LB, SPs_star)

        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            if verbose
                k += 1
                LBstr = num2str(LB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("┆Eqv.I ┆%3s ┆ %9s  ┆    ----    ┆ %8s  ┆\n", k, LBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MPs ---
        x_new = reduce(vcat, add_var.(MPs, Sx))
        π_new = reduce(vcat, add_var.(MPs, Sπ))
        δ_new = reduce(vcat, add_var.(MPs, Sδ))
        γ_new = reduce(vcat, add_var.(MPs, Sγ))
        @constraint(MPs, θ <= g * z_star + d * x_new)
        @constraint(MPs, E * x_new >= f - R * u - D * y_star - G * z_star)
        @constraint(MPs, E' * π_new <= d')
        @constraint(MPs, x_new <= BigM * δ_new)
        @constraint(MPs, d' - E' * π_new <= BigM * (1 .- δ_new))
        @constraint(MPs, π_new <= BigM * γ_new)
        @constraint(MPs, E * x_new - f + R * u + D * y_star + G * z_star <= BigM * (1 .- γ_new))
        # ------------------------------

        optimize!(MPs)
        if termination_status(SPs) == OPTIMAL
            Q_star = objective_value(MPs)
            u_star = value.(u)
        else
            if verbose
                k += 1
                term_stat = "MPs " * string(termination_status(MPs))
                @printf("┆Eqv.I ┆%3s ┆ Fail to calc. U.B.: %14s. ┆\n", k, term_stat)
            end
            break
        end
        UB = Q_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("┆Eqv.I ┆%3s ┆ %9s  ┆ %9s  ┆ %8s  ┆\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            break
        end

        # --- Update Sub-Problem (SPs) ---
        # Alternative method:
        delete(SPs, con); unregister(SPs, :con)
        @constraint(SPs, con, E * x + G * z >= f - R * u_star - D * y_star)
        # --------------------------------

    end

    return u_star, Q_star, term_stat
end




"""
    InnerLevel_CCG_EquivalentII(args..., U::Polyhedron; kwargs...)

If the extended relately complete recourse assumption does not hold, the KKT condition based reformulation (Bi/Tri-Equivalent I) is not valid. Now, the strong duality based reformulation (Bi/Tri-Equivalent II) is still valid and the corresponding C&CG variant is applicable. So here we take it as a standby method.
"""
function InnerLevel_CCG_EquivalentII(
    y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, U::Polyhedron; 
    MPsolver = Ipopt, 
    MPsolver_max_iter,
    SPsolver = HiGHS, 
    ϵ, 
    verbose
)
    
    r = fulldim(U)
    Sπ = [Rplus(length(f))]

    k = 0
    LB = -Inf; UB = +Inf
    term_stat = ""

    # --- Master Problem (MPs) Initialization ---
    MPs = Model(MPsolver.Optimizer)
    set_optimizer_attributes(MPs, "tol" => ϵ, "max_iter" => MPsolver_max_iter)
    set_silent(MPs)
    @variable(MPs, u[1:r] ∈ U)
    @variable(MPs, θ)
    @objective(MPs, Max, θ)
    # -------------------------------------------

    Q_star = +Inf
    UB = Q_star

    while true
        # --- Call Another C&CG Procedure to Solve SPs ---
        z_star, SPs_star = InnerLevel_CCG_EquivalentII_SSP(
            y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, Sπ; 
            solver = SPsolver, 
            ϵ = 1e-5, 
            verbose = true
        )
        # ------------------------------------------------
        LB = max(LB, g * z_star + SPs_star)

        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            if verbose
                k += 1
                LBstr = num2str(LB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("┆Eqv.II┆%3s ┆ %9s  ┆    ----    ┆ %8s  ┆\n", k, LBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MPs ---
        π_new = reduce(vcat, add_var.(MPs, Sπ))
        @constraint(MPs, θ <= g * z_star + (f - R * u - D * y_star - G * z_star)' * π_new)
        @constraint(MPs, E' * π_new <= d')
        # ------------------------------
        
        optimize!(MPs)
        if termination_status(MPs) == OPTIMAL
            k += 1
            u_star = value.(u)
            Q_star = objective_value(MPs)
        elseif termination_status(MPs) == LOCALLY_SOLVED
            if verbose
                term_stat = "MPs " * string(termination_status(MPs))
                k += 1
                u_star = value.(u)
                Q_star = objective_value(MPs)
                LBstr = num2str(LB, 2, 9)
                @printf("┆Eqv.II┆%3s ┆ %9s  ┆     LOCALLY_SOLVED     ┆\n", k, LBstr)
            end
            break
        else
            if verbose
                k += 1
                term_stat = "MPs " * string(termination_status(MPs))
                @printf("┆Eqv.II┆%3s ┆ Fail to calc. U.B.: %14s. ┆\n", k, term_stat)
            end
            break
        end
        UB = Q_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("┆Eqv.II┆%3s ┆ %9s  ┆ %9s  ┆ %8s  ┆\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            term_stat = "OPTIMAL"
            break
        end
    end

    return u_star, Q_star, term_stat
end





"""
    InnerLevel_CCG_EquivalentII_SSP(args...; kwargs...)

Another C&CG procedure (essentially maybe a Benders' decomposition procedure) for solving the sub-problem of the SP of the strong duality based reformulation (Bi/Tri-Equivalent II).
"""
function InnerLevel_CCG_EquivalentII_SSP(
    y_star, u_star, d, g, E, G, f, D, R, Sx, Sz, Sπ; 
    solver, 
    ϵ, 
    verbose
)
    
    p = fulldim(Sx)
    n = fulldim(Sz)

    k = 0
    LB = -Inf; UB = +Inf

    # --- Master Problem (MPss) Initialization ---
    MPss = Model(solver.Optimizer)
    set_silent(MPss)
    z = reduce(vcat, add_var.(MPss, Sz))
    x = reduce(vcat, add_var.(MPss, Sx))
    @variable(MPss, η)
    @constraint(MPss, E * x + G * z >= f - R * u_star - D * y_star)
    # --------------------------------------------

    @objective(MPss, Min, g * z)
    optimize!(MPss) 
    z_star = value.(z); η_star = -Inf
    LB = g * z_star + η_star
    @objective(MPss, Min, g * z + η)

    # --- Sub-Problem (SPss) Initialization ---
    SPss = Model(solver.Optimizer)
    set_silent(SPss)
    π = reduce(vcat, add_var.(SPss, Sπ))
    @constraint(SPss, E' * π <= d')
    @objective(SPss, Max, (f - R * u_star - D * y_star - G * z_star)' * π)
    # -----------------------------------------

    while true
        optimize!(SPss)
        if termination_status(SPss) == OPTIMAL
            SPss_star = objective_value(SPss)
            π_star = value.(π)
        else
            println(termination_status(SPss))
            # To be written.
        end
        UB = min(UB, g * z_star + SPss_star)

        if UB - LB <= ϵ
            if verbose
                k += 1
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("┆ SPs: ┆%3s ┆    ----    ┆ %9s  ┆ %8s  ┆\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MPss ---
        @constraint(MPss, η >= (f - R * u_star - D * y_star - G * z)' * π_star)
        # -------------------------------

        optimize!(MPss)
        z_star = value.(z); η_star = value.(η)
        LB = g * z_star + η_star

        k += 1
        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("┆ SPs: ┆%3s ┆ %9s  ┆ %9s  ┆ %8s  ┆\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

        # --- Update Sub-Problem (SPss) ---
        @objective(SPss, Max, (f - R * u_star - D * y_star - G * z_star)' * π)
        # --------------------------------

    end

    SPs_star = UB

    return z_star, SPs_star
end