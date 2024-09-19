"""
    SolveTwoStageRO_ECCG(args...; kwargs...)

Solve the following two-stage robust optimization problems using Extended Column-and-Constraint Generation (ECCG) method (`Ref.1`):
``` formulation
Min_x a' * x + Max_{u ∈ U} Min_{y ∈ Y(x, u)} b' * y

Subject to:

    E * x ≥ e, x ∈ Sx

    Y(x, u) = {y ∈ Sy: A * x + B * y + C * u ≥ c}

    U is the uncertainty set
```
where `Sx` is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space, and `Sy` is allowed to be a nonnegative real-valued space.

P.S. We hope the Feasibility assumption holds, i.e., there **exists** `x ∈ {x ∈ Sx: E * x ≥ e}` such that for any `u ∈ U` the set `Y(x, u) ≠ ∅`, while the Relatively Complete Recourse assumption (i.e., for **any** `x ∈ {x ∈ Sx: E * x ≥ e}` and any `u ∈ U` the set `Y(x, u) ≠ ∅`) doesn't neccessarily hold. (`Ref.1`)

# Arguments
- `a::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `a` should be converted into a vector form first, e.g., `a = [2]`.
- `b::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `b` should be converted into a vector form first, e.g., `b = [1]`.
- `E::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `E` should be converted into a matrix form first, e.g., `E = [1 2]`, `E = [1; 2;;]` or `E = [0;;]`.
- `e::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `e` should be converted into a vector form first, e.g., `e = [0]`.
- `B::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `B` should be converted into a matrix form first, e.g., `B = [1 2]`, `B = [1; 2;;]` or `B = [1;;]`.
- `c::AbstractVector{<:Real}`: It should be a vector. A degenerate scalar `c` should be converted into a vector form first, e.g., `c = [0]`.
- `A::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `A` should be converted into a matrix form first, e.g., `A = [1 2]`, `A = [1; 2;;]` or `A = [-1;;]`.
- `C::AbstractMatrix{<:Real}`: It should be a matrix. A degenerate vector or scalar `C` should be converted into a matrix form first, e.g., `C = [1 2]`, `C = [1; 2;;]` or `C = [-1;;]`.
- `Sx::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space. It should be constructed as a vector consisting of (a mixture of) instances of the following three basic data types: `Rplus`, `ZeroOne` and `Zplus`. E.g., `Sy = [ZeroOne()]` denotes an one-dimensional binary space `{0, 1}` and `Sy = [Rplus(2), Zplus(3)]` denotes a five-dimensianal nonnegative mixed integer space `ℝ₊² × ℤ₊³`. Note that the latter is equivalent to `Sy = [fill(Rplus(), 2), fill(Zplus(), 3)]`.
- `Sy::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space. The construction method is similar to `Sx`.
- `U::Union{Model, Polyhedron}`: The uncertainty set is allowed to be constructed as a `Model` using package `JuMP.jl` or a `Polyhedron` using package `Polyhedra.jl`. This package has integrated some neccessary functions from `JuMP.jl`, `Polyhedra.jl` and some other related packages, which allows the user to model the uncertainty set conveniently without having to additionally `using/import` the above packages. Please see the `Examples` section for brief usage. Note that only the variables whose name is registered as `u` in a `JuMP` model uncertainty set will be recognized as the uncertainty variables unless there is only one registered variable name (or all variables are anonymous).

# Keywords
- `MPsolver::Module = HiGHS`: The solver for the master problem (`MP`). It should be chosen based on the tpyes of `Sx` and `Sy`. Generally, it should be an solver that supports mixed integer linear program (MILP) if `Sx` or `Sy` is a mixed integer space.
- `SPsolver::Module = HiGHS`: The solver for the sub-problem (`SP`) in the KKT condition based reformulation form (both the feasiblity oracle and the optimality oracle). Since the sub-problem oracles are linearized reformulations based on big-M method, their solver should at least be an MILP solver, if `U` is just a polyhedral uncertainty set.
- `ϵ::Real = 1e-5`: The overall absolute stopping criteria of the Extended CCG method. It's also used as the tolorence in the feasiblity oracle for which if the objective value of the oracle is not less than `ϵ` we then think the current `x` is infeasible.
- `BigM::Real = 1e5`: The big-M value of `SP` in its big-M method based linearized reformulation (both the feasiblity oracle and the optimality oracle). Note that if a tight bound on big-M can be analytically obtained, a better performance of the algorithm can be achieved.
- `verbose::Bool = true`: The switch that controls the output of the solution process details.

# Returns
This function will return a `Tuple{Vector{Real}, Real, Integer}` whose entries are in order:
- `x::Vector{Real}`: the optimal solution to the first stage decision variable `x`.
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
y, objv = SolveTwoStageRO_ECCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
# # Call the function with a `Polyhedron` type uncertianty set:
# UncSet = polyhedron(HalfSpace([1], 1) ∩ HalfSpace([-1], 0))
# y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
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
y, objv = SolveTwoStageRO_ECCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
# # Call the function with a `Polyhedron` type uncertianty set:
# UncSet = polyhedron(UncMod, CDDLib.Library(:exact))
# ind_δ = ind_of_var(UncMod, "δ")
# UncSet = eliminate(UncSet, ind_δ)
# y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
```

# References
1. Bertsimas, D., & Shtern, S. (2018). A scalable algorithm for two-stage adaptive linear optimization. arXiv preprint arXiv:1807.02812.

"""
function SolveTwoStageRO_ECCG end

############################################################################
#                       Sy::Vector{Rplus} & U::Model                       #
############################################################################

function SolveTwoStageRO_ECCG(
    a::AbstractVector{<:Real}, 
    b::AbstractVector{<:Real}, 
    E::AbstractMatrix{<:Real}, 
    e::AbstractVector{<:Real}, 
    B::AbstractMatrix{<:Real}, 
    c::AbstractVector{<:Real}, 
    A::AbstractMatrix{<:Real}, 
    C::AbstractMatrix{<:Real}, 
    Sx::Vector{<:BasicDomain}, 
    Sy::Vector{Rplus}, 
    U::Model; 
    MPsolver::Module = HiGHS, 
    SPsolver::Module = HiGHS, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true
)

    n = fulldim(Sx); m = fulldim(Sy); l = length(get_uncertainty_variable(U)); r = length(c)

    k = 0
    LB = -Inf; UB = +Inf

    # --- Get the First u_star ---
    UncMod = copy(U)
    set_optimizer(UncMod, SPsolver.Optimizer)
    set_silent(UncMod)
    u0 = get_uncertainty_variable(UncMod)
    @objective(UncMod, Max, sum(u0))
    optimize!(UncMod)
    u_star = value.(u0)
    # ----------------------------

    # --- Master Problem Initialization ---
    MP = Model(MPsolver.Optimizer)
    set_silent(MP)
    x = reduce(vcat, add_var.(MP, Sx))
    @variable(MP, θ)
    @constraint(MP, E * x >= e)
    y_new = reduce(vcat, add_var.(MP, Sy))
    @constraint(MP, θ >= b' * y_new)
    @constraint(MP, A * x + B * y_new + C * u_star >= c)
    @objective(MP, Min, a' * x + θ)
    # -------------------------------------

    # --- Sub-Problem Initialization ---
    x_star = zeros(n) # dummy value

    # Feasibility Oracle:
    SP_fea = copy(U)
    set_optimizer(SP_fea, SPsolver.Optimizer)
    set_silent(SP_fea)
    u_fea = get_uncertainty_variable(SP_fea)
    y_fea = reduce(vcat, add_var.(SP_fea, Sy))
    γ_fea = @variable(SP_fea; lower_bound = 0)
    w_fea = @variable(SP_fea, [1:r]; lower_bound = 0)
    α_fea = @variable(SP_fea, [1:r]; binary = true)
    β_fea = @variable(SP_fea, [1:m+1]; binary = true)
    @constraint(SP_fea, con1_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) >= c)
    @constraint(SP_fea, [B ones(r)]' * w_fea <= [zeros(m); 1])
    @constraint(SP_fea, w_fea <= BigM * α_fea)
    @constraint(SP_fea, con2_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) - c <= BigM * (1 .- α_fea))
    @constraint(SP_fea, [y_fea; γ_fea] <= BigM * β_fea)
    @constraint(SP_fea, [zeros(m); 1] - [B ones(r)]' * w_fea <= BigM * (1 .- β_fea))
    @objective(SP_fea, Max, γ_fea)

    # Optimality Oracle:
    SP_opt = copy(U)
    set_optimizer(SP_opt, SPsolver.Optimizer)
    set_silent(SP_opt)
    u_opt = get_uncertainty_variable(SP_opt)
    y_opt = reduce(vcat, add_var.(SP_opt, Sy))
    w_opt = @variable(SP_opt, [1:r]; lower_bound = 0)
    α_opt = @variable(SP_opt, [1:r]; binary = true)
    β_opt = @variable(SP_opt, [1:m]; binary = true)
    @constraint(SP_opt, con1_opt, A * x_star + B * y_opt + C * u_opt >= c)
    @constraint(SP_opt, B' * w_opt <= b)
    @constraint(SP_opt, w_opt <= BigM * α_opt)
    @constraint(SP_opt, con2_opt, A * x_star + B * y_opt + C * u_opt - c <= BigM * (1 .- α_opt))
    @constraint(SP_opt, y_opt <= BigM * β_opt)
    @constraint(SP_opt, b - B' * w_opt <= BigM * (1 .- β_opt))
    @objective(SP_opt, Max, b' * y_opt)
    # ----------------------------------

    if verbose
        @printf("|-------------- Extended-CCG Method --------------|\n")
        @printf("|   Iter.   |    L.B.    |    U.B.    |    Gap    |\n")
        @printf("|-----------|------------|------------|-----------|\n")
        LBstr = num2str(LB, 2, 9)
        UBstr = num2str(UB, 2, 9)
        GAPstr = num2str(UB - LB, 2, 8) 
        @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
    end

    while true
        k = k + 1

        # --- Solve the Master Problem ---
        optimize!(MP)
        if ~is_solved_and_feasible(MP)
            error("The Master Problem is not solved or it's infeasible!")
        end
        x_star = value.(x); θ_star = value.(θ)
        LB = a' * x_star + θ_star
        # --------------------------------

        if UB - LB <= ϵ
            if verbose
                LBstr = num2str(LB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    | %9s  |    ----    | %8s  |\n", k, LBstr, GAPstr)
            end
            break
        end

        # --- Solve the Sub-Problem ---
        # ----- Update SP_fea ------
        delete(SP_fea, con1_fea); unregister(SP_fea, :con1_fea)
        delete(SP_fea, con2_fea); unregister(SP_fea, :con2_fea)
        @constraint(SP_fea, con1_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) >= c)
        @constraint(SP_fea, con2_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) - c <= BigM * (1 .- α_fea))
        # --------------------------
        optimize!(SP_fea) # Feasibility Check
        if termination_status(SP_fea) == OPTIMAL
            if value(γ_fea) >= ϵ # x_star is infeasible.
                UB = +Inf
                u_star = value.(u_fea)
            else # x_star is feasible
                # ----- Update SP_opt ------
                delete(SP_opt, con1_opt); unregister(SP_opt, :con1_opt)
                delete(SP_opt, con2_opt); unregister(SP_opt, :con2_opt)
                @constraint(SP_opt, con1_opt, A * x_star + B * y_opt + C * u_opt >= c)
                @constraint(SP_opt, con2_opt, A * x_star + B * y_opt + C * u_opt - c <= BigM * (1 .- α_opt))
                # --------------------------
                optimize!(SP_opt)
                if termination_status(SP_opt) == OPTIMAL
                    UB = a' * x_star + objective_value(SP_opt)
                    u_star = value.(u_opt)
                else
                    error("Optimality Oracle Exception!")
                end
            end
        else
            error("Feasibility Oracle Exception!")
        end
        # -----------------------------

        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

        # --- Add Constraints to MP ---
        y_new = reduce(vcat, add_var.(MP, Sy))
        @constraint(MP, θ >= b' * y_new)
        @constraint(MP, A * x + B * y_new + C * u_star >= c)
        # -----------------------------
    end

    objv = UB

    return value.(x), objv, k
end


############################################################################
#                    Sy::Vector{Rplus} & U::Polyhedron                     #
############################################################################

function SolveTwoStageRO_ECCG(
    a::AbstractVector{<:Real}, 
    b::AbstractVector{<:Real}, 
    E::AbstractMatrix{<:Real}, 
    e::AbstractVector{<:Real}, 
    B::AbstractMatrix{<:Real}, 
    c::AbstractVector{<:Real}, 
    A::AbstractMatrix{<:Real}, 
    C::AbstractMatrix{<:Real}, 
    Sx::Vector{<:BasicDomain}, 
    Sy::Vector{Rplus}, 
    U::Polyhedron; 
    MPsolver::Module = HiGHS, 
    SPsolver::Module = HiGHS, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true
)

    n = fulldim(Sx); m = fulldim(Sy); l = fulldim(U); r = length(c)

    k = 0
    LB = -Inf; UB = +Inf

    # --- Get the First u_star ---
    UncMod = Model(SPsolver.Optimizer)
    set_silent(UncMod)
    @variable(UncMod, u0[1:l] ∈ U)
    @objective(UncMod, Max, sum(u0))
    optimize!(UncMod)
    u_star = value.(u0)
    # ----------------------------

    # --- Master Problem Initialization ---
    MP = Model(MPsolver.Optimizer)
    set_silent(MP)
    x = reduce(vcat, add_var.(MP, Sx))
    @variable(MP, θ)
    @constraint(MP, E * x >= e)
    y_new = reduce(vcat, add_var.(MP, Sy))
    @constraint(MP, θ >= b' * y_new)
    @constraint(MP, A * x + B * y_new + C * u_star >= c)
    @objective(MP, Min, a' * x + θ)
    # -------------------------------------

    # --- Sub-Problem Initialization ---
    x_star = zeros(n) # dummy value

    # Feasibility Oracle:    
    SP_fea = Model(SPsolver.Optimizer)
    set_silent(SP_fea)
    @variable(SP_fea, u_fea[1:l] ∈ U)
    y_fea = reduce(vcat, add_var.(SP_fea, Sy))
    γ_fea = @variable(SP_fea; lower_bound = 0)
    w_fea = @variable(SP_fea, [1:r]; lower_bound = 0)
    α_fea = @variable(SP_fea, [1:r]; binary = true)
    β_fea = @variable(SP_fea, [1:m+1]; binary = true)
    @constraint(SP_fea, con1_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) >= c)
    @constraint(SP_fea, [B ones(r)]' * w_fea <= [zeros(m); 1])
    @constraint(SP_fea, w_fea <= BigM * α_fea)
    @constraint(SP_fea, con2_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) - c <= BigM * (1 .- α_fea))
    @constraint(SP_fea, [y_fea; γ_fea] <= BigM * β_fea)
    @constraint(SP_fea, [zeros(m); 1] - [B ones(r)]' * w_fea <= BigM * (1 .- β_fea))
    @objective(SP_fea, Max, γ_fea)

    # Optimality Oracle:
    SP_opt = Model(SPsolver.Optimizer)
    set_silent(SP_opt)
    @variable(SP_opt, u_opt[1:l] ∈ U)
    y_opt = reduce(vcat, add_var.(SP_opt, Sy))
    w_opt = @variable(SP_opt, [1:r]; lower_bound = 0)
    α_opt = @variable(SP_opt, [1:r]; binary = true)
    β_opt = @variable(SP_opt, [1:m]; binary = true)
    @constraint(SP_opt, con1_opt, A * x_star + B * y_opt + C * u_opt >= c)
    @constraint(SP_opt, B' * w_opt <= b)
    @constraint(SP_opt, w_opt <= BigM * α_opt)
    @constraint(SP_opt, con2_opt, A * x_star + B * y_opt + C * u_opt - c <= BigM * (1 .- α_opt))
    @constraint(SP_opt, y_opt <= BigM * β_opt)
    @constraint(SP_opt, b - B' * w_opt <= BigM * (1 .- β_opt))
    @objective(SP_opt, Max, b' * y_opt)
    # ----------------------------------

    if verbose
        @printf("|-------------- Extended-CCG Method --------------|\n")
        @printf("|   Iter.   |    L.B.    |    U.B.    |    Gap    |\n")
        @printf("|-----------|------------|------------|-----------|\n")
        LBstr = num2str(LB, 2, 9)
        UBstr = num2str(UB, 2, 9)
        GAPstr = num2str(UB - LB, 2, 8) 
        @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
    end

    while true
        k = k + 1

        # --- Solve the Master Problem ---
        optimize!(MP)
        if ~is_solved_and_feasible(MP)
            error("The Master Problem is not solved or it's infeasible!")
        end
        x_star = value.(x); θ_star = value.(θ)
        LB = a' * x_star + θ_star
        # --------------------------------

        if UB - LB <= ϵ
            if verbose
                LBstr = num2str(LB, 2, 9)
                GAPstr = num2str(UB - LB, 2, 8) 
                @printf("| %6s    | %9s  |    ----    | %8s  |\n", k, LBstr, GAPstr)
            end
            break
        end

        # --- Solve the Sub-Problem ---
        # ----- Update SP_fea ------
        delete(SP_fea, con1_fea); unregister(SP_fea, :con1_fea)
        delete(SP_fea, con2_fea); unregister(SP_fea, :con2_fea)
        @constraint(SP_fea, con1_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) >= c)
        @constraint(SP_fea, con2_fea, A * x_star + B * y_fea + C * u_fea + γ_fea * ones(r) - c <= BigM * (1 .- α_fea))
        # --------------------------
        optimize!(SP_fea) # Feasibility Check
        if termination_status(SP_fea) == OPTIMAL
            if value(γ_fea) >= ϵ # x_star is infeasible.
                UB = +Inf
                u_star = value.(u_fea)
            else # x_star is feasible
                # ----- Update SP_opt ------
                delete(SP_opt, con1_opt); unregister(SP_opt, :con1_opt)
                delete(SP_opt, con2_opt); unregister(SP_opt, :con2_opt)
                @constraint(SP_opt, con1_opt, A * x_star + B * y_opt + C * u_opt >= c)
                @constraint(SP_opt, con2_opt, A * x_star + B * y_opt + C * u_opt - c <= BigM * (1 .- α_opt))
                # --------------------------
                optimize!(SP_opt)
                if termination_status(SP_opt) == OPTIMAL
                    UB = a' * x_star + objective_value(SP_opt)
                    u_star = value.(u_opt)
                else
                    error("Optimality Oracle Exception!")
                end
            end
        else
            error("Feasibility Oracle Exception!")
        end
        # -----------------------------

        if verbose
            LBstr = num2str(LB, 2, 9)
            UBstr = num2str(UB, 2, 9)
            GAPstr = num2str(UB - LB, 2, 8) 
            @printf("| %6s    | %9s  | %9s  | %8s  |\n", k, LBstr, UBstr, GAPstr)
        end
        if UB - LB <= ϵ
            break
        end

        # --- Add Constraints to MP ---
        y_new = reduce(vcat, add_var.(MP, Sy))
        @constraint(MP, θ >= b' * y_new)
        @constraint(MP, A * x + B * y_new + C * u_star >= c)
        # -----------------------------
    end

    objv = UB

    return value.(x), objv, k
end