############################################################################
#                                 Main UI                                  #
############################################################################


"""
    solve(model::TSROModel; algorithm::TSRO_Algorithm) -> y_star, objv, recourse

Solve the two-satage robust optimization model `model` with algorithm `algorithm`.

# Arguments
- `model::TSROModel`: the two-satage robust optimization model to solve.

# Keywords
- `algorithm::TSRO_Algorithm`: the solving algorithm.

# Returns
- `y_star`: the first stage optimal decision.
- `objv`: the optimal objective value of the two-stage robust optimization problem.
- `recourse`: the recourse function for getting the second stage optimal decision and the the optimal objective value of the recourse problem when the uncertainty reveals, i.e., recourse(û) -> x_star, RPobjv.

# Examples

### Case-1: 
``` julia
using MKLTwoStageRO

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

### Case-2: 
``` julia
using MKLTwoStageRO

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

### Case-3: 
``` julia
using MKLTwoStageRO

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
eccg = ECCG()
y_star, objv, recourse = solve(model; algorithm=eccg)
û = [0.5, -0.5]
x_star, RPobjv = recourse(û)
```

### Case-4: 
``` julia
using MKLTwoStageRO
# if the user wants parallelization, replace it with the following code:
# using Distributed # if the user wants parallelization
# addprocs(5)
# @everywhere using MKLTwoStageRO
using GLMakie # if the user wants visualization

# Uncertain Data Generation
u1 = 0.5 .+ 4 * rand(300)
u2 = 2 ./ u1 + 0.3 * randn(300) .+ 1
X = [u1'; u2']

# MKL-based Uncertainty Set Construction
algor = HessianMKL(verbose=false)
U = mklocsvmtrain(X, 21; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05)
# U = mklocsvmtrain(X, 50; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05, num_batch=5) # if the user wants parallelization
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
model = TSROModel(a, b, E, e, B, c, A, C, Sx, Sy, U)

# Solve the Model Using MKLCCG Algorithm
mklccg = MKLCCG()
x_star, objv, recourse = solve(model; algorithm=mklccg)

# Retrieve the Recourse Decision After the Uncertainty Reveals
û = [2.0, 3.0] # assume this is the uncertainty observation
y_star, RPobjv = recourse(û)
```
"""
function solve(model::TSROModel; algorithm::TSRO_Algorithm)
    y_star, objv, _ = algorithm(model)

    function recourse(û)
        RP = Model(algorithm.MPsolver.Optimizer)
        set_silent(RP)
        x = reduce(vcat, add_var.(RP, model.Sx))
        @constraint(RP, model.G * x >= model.h - model.E * y_star - model.M * û)
        @objective(RP, Min, model.b' * x)
        optimize!(RP)
        x_star = value.(x)
        RPobjv = objective_value(RP)
        return x_star, RPobjv
    end

    y_star, objv, recourse
end





############################################################################
#                             Uncertainty Set                              #
############################################################################
"Extract the indices of a certain variable named as `x` from the vector `all_variables(model)` of a JuMP model `model`. "
function ind_of_var(model::Model, x::String)
    indices = []
    for (i, var) in enumerate(all_variables(model))
        if name(var) == x || startswith(name(var), x * "[")
            push!(indices, i)
        end
    end
    return indices
end

"Get the uncertainty variable from the uncertainty set built as a `JuMP` model. If the model has a single registered variable name (or the single anonymous `_`) it will be considered as the uncertainty variable, otherwise only the variable whose name is registered as `u` will be considered as the uncertainty variable."
function get_uncertainty_variable(model::Model)
    try
        (model[:u] isa AbstractArray) ? u = model[:u] : u = [model[:u]]
    catch
        num_reg_var_names = length(collect(keys(object_dictionary(model))))
        num_reg_vars = sum(length.(collect(values(object_dictionary(model)))))
        num_vars = num_variables(model)
        if num_reg_var_names <= 1 && num_reg_vars == num_vars
            u = all_variables(model)
        else
            error("Uncertainty set model not constructed legally: If the model has a single registered variable name (or the single anonymous `_`) it will be considered as the uncertainty variable, otherwise only the variable whose name is registered as `u` will be considered as the uncertainty variable.")
        end
    end
end






############################################################################
#                                  Others                                  #
############################################################################

"Convert a real number to a string with its width less than `max_width`."
function num2str(num::Real, digits::Integer, max_width::Integer)
    str = format(Format("%." * string(digits) * "f"), num)
    if length(str) > max_width
        str = format(Format("%." * string(digits) * "e"), num)
    end
    return str
end

