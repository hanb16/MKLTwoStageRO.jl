
"""
    SolveTwoStageRO_BDCP(args...; kwargs...)

Solve the following two-stage robust optimization problems using Benders-dual cutting plane (BDCP) method mentioned in `Ref.1`:
``` formulation
Min_y c' * y + Max_{u ∈ U} Min_{x ∈ F(y, u)} b' * x

Subject to:

    A * y ≥ d, y ∈ Sy

    F(y, u) = {x ∈ Sx: G * x ≥ h - E * y - M * u}

    U is the uncertainty set
```
where `Sy` is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space, and `Sx` is allowed to be a nonnegative real-valued space.

P.S. We hope the Relatively Complete Recourse assumption holds, i.e., the second-stage decision problem (which is a linear program) is feasible for any given `y` and `u`. (`Ref.1`)

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
- `Sx::Vector{Rplus}`: It should be a nonnegative real-valued space that is constructed as a vector consisting of instances of basic data type `Rplus`, e.g., `Sx = [Rplus(3)]` or equivalently `Sx = fill(Rplus(), 3)`.
- `U::Union{Model, Polyhedron}`: The uncertainty set is allowed to be constructed as a `Model` using package `JuMP.jl` or a `Polyhedron` using package `Polyhedra.jl`. This package has integrated some neccessary functions from `JuMP.jl`, `Polyhedra.jl` and some other related packages, which allows the user to model the uncertainty set conveniently without having to additionally `using/import` the above packages. Please see the `Examples` section for brief usage. Note that only the variables whose name is registered as `u` in a `JuMP` model uncertainty set will be recognized as the uncertainty variables unless there is only one registered variable name (or all variables are anonymous).

# Keywords
- `MPsolver::Module = HiGHS`: The solver for the master problem (`MP`). It should be chosen based on the tpye of `Sy`. Generally, it should be an solver that supports mixed integer linear program (MILP) if `Sy` is a mixed integer space. 
- `SP1solver::Module = HiGHS`: The solver for the sub-problem in the KKT condition based reformulation (Bi/Tri-Equivalent I) form (`SP1`). Since `SP1` is a linearized reformulation based on big-M method, its solver should at least be an MILP solver, if `U` is just a polyhedral uncertainty set.
- `SP2solver::Module = Ipopt`: The solver for the sub-problem in the strong duality based reformulation form (`SP2`). The function will always try to solve `SP1` first, and then move on to `SP2` if it fails. Since `SP2` is at least a bilinear optimization problem when `U` is a polyhedron, a solver that is tailored to this or roughly one that supports non-convex quadratic program (QP) is neccessary. 
- `SP2solver_max_iter::Integer = 10000`: The maximum limitation of iterations for `SP2solver` to solve `SP2`.
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
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncMod) 
# Call the function with a `Polyhedron` type uncertianty set:
UncSet = polyhedron(HalfSpace([1], 1) ∩ HalfSpace([-1], 0))
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
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
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
# Call the function with a `Polyhedron` type uncertianty set:
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))
ind_δ = ind_of_var(UncMod, "δ")
UncSet = eliminate(UncSet, ind_δ)
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
```

# Reference
1. Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.
"""
function SolveTwoStageRO_BDCP end

############################################################################
#                        For General Uncertainty Set                       #
############################################################################

function SolveTwoStageRO_BDCP(
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
        @printf("|------- Benders-Dual Cutting Plane Method -------|\n")
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
            π_star = value.(π1)
        else
            optimize!(SP2)
            if termination_status(SP2) in [OPTIMAL, LOCALLY_SOLVED, ITERATION_LIMIT, ALMOST_LOCALLY_SOLVED, NORM_LIMIT, NUMERICAL_ERROR]
                Q_y_star = objective_value(SP2)
                u_star = value.(u2)
                π_star = value.(π2)
            else
                error("SP2 Solution Exception!")
            end
        end
        
        UB = min(UB, c' * y_star + Q_y_star)
        
        if UB - LB <= ϵ
            k += 1
            if verbose
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    |    ----    | %9s  | %8s  |\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MP ---
        # @constraint(MP, η >= (h - E * y - M * u_star)' * π_star)
        while true
            try
                @constraint(MP, η >= (h - E * y - M * u_star)' * π_star)
                break
            catch
                π_star = π_star / 100
            end
        end
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



############################################################################
#                      For Polyhedral Uncertainty Set                      #
############################################################################

function SolveTwoStageRO_BDCP(
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
        @printf("|------- Benders-Dual Cutting Plane Method -------|\n")
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
    @variable(SP1, u1[1:r] ∈ U)
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
    SP2 = Model(SP2solver.Optimizer)
    set_optimizer_attributes(SP2, "tol" => ϵ, "max_iter" => SP2solver_max_iter)
    set_silent(SP2)
    @variable(SP2, u2[1:r] ∈ U)
    π2 = @variable(SP2, [eachindex(h)]; lower_bound = 0)
    @constraint(SP2, G' * π2 <= b)
    @objective(SP2, Max, (h - E * y_star - M * u2)' * π2)
    # ----------------------------------

    while true
        optimize!(SP1)
        if termination_status(SP1) == OPTIMAL
            Q_y_star = objective_value(SP1)
            u_star = value.(u1)
            π_star = value.(π1)
        else
            optimize!(SP2)
            if termination_status(SP2) in [OPTIMAL, LOCALLY_SOLVED, ITERATION_LIMIT, ALMOST_LOCALLY_SOLVED, NORM_LIMIT, NUMERICAL_ERROR]
                Q_y_star = objective_value(SP2)
                u_star = value.(u2)
                π_star = value.(π2)
            else
                error("SP2 Solution Exception!")
            end
        end
        
        UB = min(UB, c' * y_star + Q_y_star)
        
        if UB - LB <= ϵ
            k += 1
            if verbose
                UBstr = num2str(UB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    |    ----    | %9s  | %8s  |\n", k, UBstr, GAPstr)
            end
            break
        end

        # --- Add Constraints to MP ---
        # @constraint(MP, η >= (h - E * y - M * u_star)' * π_star)
        while true
            try
                @constraint(MP, η >= (h - E * y - M * u_star)' * π_star)
                break
            catch
                π_star = π_star / 100
            end
        end
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