
"""
    SolveTwoStageRO_MKLCCG(args...; kwargs...)

Solve the following two-stage robust optimization problems using Multiple Kernel Learning-aided Colum-and-Constraint Generation (MKLCCG) method:
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
- `Sx::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space or a nonnegative mixed integer space. It should be constructed as a vector consisting of (a mixture of) instances of the following three basic data types: `Rplus`, `ZeroOne` and `Zplus`. E.g., `Sx = [ZeroOne()]` denotes an one-dimensional binary space `{0, 1}` and `Sx = [Rplus(2), Zplus(3)]` denotes a five-dimensianal nonnegative mixed integer space `ℝ₊² × ℤ₊³`. Note that the latter is equivalent to `Sx = [fill(Rplus(), 2), fill(Zplus(), 3)]`.
- `Sy::Vector{<:BasicDomain}`: It is allowed to be a nonnegative real-valued space. The construction method is similar to `Sx`.
- `U::Union{StandardModel, CombinedModel, Vector{StandardModel}}`: The uncertainty set is allowed to be constructed as a `StandardModel`, a `CombinedModel` or a `Vector{StandardModel}` using package [`MKLOneClassSVM.jl`](https://github.com/hanb16/MKLOneClassSVM.jl). This package has integrated some neccessary functions from `MKLOneClassSVM.jl`, which allows the user to model the uncertainty set conveniently without having to additionally `using/import` this package. Please see the `Examples` section for brief usage. 

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

### Case-1:
``` julia
using MKLTwoStageRO
using GLMakie # if the user wants visualization

# Uncertain Data Generation
u1 = 0.5 .+ 4 * rand(300)
u2 = 2 ./ u1 + 0.3 * randn(300) .+ 1
X = [u1'; u2']

# MKL-based Uncertainty Set Construction
algor = HessianMKL(verbose=false)
U = mklocsvmtrain(X, 21; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05)
mklocsvmplot(U; backend=GLMakie) # visualize the MKL uncertainty set

# Instantiate the Two-Stage RO Model
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

# Call the Function to Solve the Model
y, objv = SolveTwoStageRO_MKLCCG(a, b, E, e, B, c, A, C, Sx, Sy, U)
```

### Case-2:
``` julia
using Distributed # if the user wants parallelization
addprocs(5)
@everywhere using MKLTwoStageRO
using GLMakie # if the user wants visualization

# Uncertain Data Generation
u1 = 0.5 .+ 4 * rand(300)
u2 = 2 ./ u1 + 0.3 * randn(300) .+ 1
X = [u1'; u2']

# MKL-based Uncertainty Set Construction
algor = HessianMKL(verbose=false)
U = mklocsvmtrain(X, 50; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05, num_batch=5)
mklocsvmplot(U; backend=GLMakie) # visualize the MKL uncertainty set

# Instantiate the Two-Stage RO Model
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

# Call the Function to Solve the Model
y, objv = SolveTwoStageRO_MKLCCG(a, b, E, e, B, c, A, C, Sx, Sy, U)
```

# References
1. Han, B. (2024). Multiple kernel learning-aided column-and-constraint generation method. 
2. Han, B., Shang, C., & Huang, D. (2021). Multiple kernel learning-aided robust optimization: Learning algorithm, computational tractability, and usage in multi-stage decision-making. European Journal of Operational Research, 292(3), 1004-1018.

"""
function SolveTwoStageRO_MKLCCG end


############################################################################
#                  Sy::Vector{Rplus} & U::StandardModel                    #
############################################################################

function SolveTwoStageRO_MKLCCG(
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
    U::StandardModel; 
    MPsolver::Module = HiGHS, 
    SPsolver::Module = HiGHS, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true
)

    n = fulldim(Sx); m = fulldim(Sy); l = size(U.data, 1); r = length(c)
    UM = convert_to_jumpmodel(U; form="linear")

    k = 0
    LB = -Inf; UB = +Inf

    # --- Get the First u_star ---
    UncMod = copy(UM)
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
    @objective(MP, Min, a' * x + θ)

    y_new = reduce(vcat, add_var.(MP, Sy))
    @constraint(MP, θ >= b' * y_new)
    @constraint(MP, A * x + B * y_new + C * u_star >= c)

    if ~isempty(U.BSV) # If there are BSVs, add their corresponding cut to the MP.
        for u_star in eachcol(U.data_bsv)
            y_new = reduce(vcat, add_var.(MP, Sy))
            @constraint(MP, θ >= b' * y_new)
            @constraint(MP, A * x + B * y_new + C * u_star >= c)
        end
    end
    # ------------------------------

    # --- Sub-Problem Initialization ---
    x_star = zeros(n) # dummy value

    # Feasibility Oracle:
    SP_fea = copy(UM)
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
    SP_opt = copy(UM)
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
        @printf("|--- Multiple Kernel Learning-aided CCG Method ---|\n")
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
#                  Sy::Vector{Rplus} & U::CombinedModel                    #
############################################################################

function SolveTwoStageRO_MKLCCG(
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
    U::CombinedModel; 
    MPsolver::Module = HiGHS, 
    SPsolver::Module = HiGHS, 
    ϵ::Real = 1e-5, 
    BigM::Real = 1e5, 
    verbose::Bool = true
)

    n = fulldim(Sx); m = fulldim(Sy); l = size(U.data, 1); r = length(c)
    UCM = convert_to_jumpmodel(U; form="linear")

    k = 0
    LB = -Inf; UB = +Inf

    # --- Get the First u_star ---
    UncMod = copy(UCM)
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
    @objective(MP, Min, a' * x + θ)

    y_new = reduce(vcat, add_var.(MP, Sy))
    @constraint(MP, θ >= b' * y_new)
    @constraint(MP, A * x + B * y_new + C * u_star >= c)

    if ~isempty(U.comb_BSV) # If there are comb_BSVs, add their corresponding cut to the MP.
        for u_star in eachcol(U.comb_data_bsv)
            y_new = reduce(vcat, add_var.(MP, Sy))
            @constraint(MP, θ >= b' * y_new)
            @constraint(MP, A * x + B * y_new + C * u_star >= c)
        end
    end
    # ------------------------------

    # --- Sub-Problem Initialization ---
    x_star = zeros(n) # dummy value

    # Feasibility Oracle:
    SP_fea = copy(UCM)
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
    SP_opt = copy(UCM)
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
        @printf("|--- Multiple Kernel Learning-aided CCG Method ---|\n")
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
#               Sy::Vector{Rplus} & U::Vector{StandardModel}               #
############################################################################

function SolveTwoStageRO_MKLCCG(
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
    U::Vector{StandardModel}; 
    kwargs...
)
    U = mklocsvmbuild(U)
    SolveTwoStageRO_MKLCCG(a, b, E, e, B, c, A, C, Sx, Sy, U; kwargs...)
end