include("myFuncs.jl")


############################################################################
#                          Basic Algorithm Test                            #
############################################################################

## --- Case-1: Small Numerical Instance ---
println("Case-1: Small Numerical Instance")
c = [2]; b = [1];
A = [0;;]; d = [0]; 
G = [1;;]; h = [0]; E = [-1;;]; M = [-1;;];
Sy = [ZeroOne()]; Sx = [Rplus()]; 
UncMod = Model()
@variable(UncMod, 0 <= δ <= 1)
UncSet = polyhedron(HalfSpace([1], 1) ∩ HalfSpace([-1], 0))

y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test y == [0.0] && objv == 1.0)
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test y == [0.0] && objv == 1.0)
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test isapprox(y, [0.0]; atol = 1e-5) && isapprox(objv, 1.0; atol = 1e-5))
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test isapprox(y, [0.0]; atol = 1e-5) && isapprox(objv, 1.0; atol = 1e-5))
y, objv = SolveTwoStageRO_ECCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test y == [0.0] && objv == 1.0)
y, objv = SolveTwoStageRO_ECCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test y == [0.0] && objv == 1.0)
## ----------------------------------------




## --- Case-2: Bigger Numerical Instance ---
println("Case-2: Bigger Numerical Instance")
c = [1, 2]; b = [3];
A = [4 5]; d = [6]; 
G = [1;;]; h = [2]; E = [-3 -4]; M = [-5 -6];
Sx = [Rplus()]; Sy = [Rplus(), ZeroOne()];
UncMod = Model()
u0 = [0, 0]
@variable(UncMod, u[1:2])
@variable(UncMod, -1 <= δ[1:2] <= 1)
@constraint(UncMod, u == u0 + δ)
ind_δ = ind_of_var(UncMod, "δ")
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))
UncSet = eliminate(UncSet, ind_δ)

y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test y == [1.5, 0] && objv == 54)
y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test y == [1.5, 0] && objv == 54)
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test y == [1.5, 0] && isapprox(objv, 54; atol = 1e-5))
y, objv = SolveTwoStageRO_BDCP(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test y == [1.5, 0] && isapprox(objv, 54; atol = 1e-5))
y, objv = SolveTwoStageRO_ECCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test y == [1.5, 0] && objv == 54)
y, objv = SolveTwoStageRO_ECCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test y == [1.5, 0] && objv == 54)
## -----------------------------------------




## --- Case-3: SMALL Two-Stage Robust Location-Transportation Problem ---
# See: Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.

println("Case-3: SMALL Two-Stage Robust Location-Transportation Problem")

# Model Parameter Standardization
f = [1, 2]
a = [1, 2]
c = [1; 2]
K = fill(3,2)
Params = TSRLTP_standardize(f, a, c, K)

# Uncertainty Set Construction
UncMod = Model()
@variable(UncMod, 3 <= d <= 5)
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))

# Invokating the Algorithms
y, objv = SolveTwoStageRO_CCG(Params..., UncMod)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("JuMP Model Uncertianty Set: ")
display(@test (y, objv) == ([1.0, 1.0, 3.0, 2.0], 17.0))

y, objv = SolveTwoStageRO_CCG(Params..., UncSet)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("Polyhedron Model Uncertianty Set: ")
display(@test (y, objv) == ([1.0, 1.0, 3.0, 2.0], 17.0))

y, objv = SolveTwoStageRO_BDCP(Params..., UncMod)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("JuMP Model Uncertianty Set: ")
display(@test isapprox(y, [1.0, 1.0, 3.0, 2.0]; atol = 1e-5) && isapprox(objv, 17.0; atol = 1e-5))

y, objv = SolveTwoStageRO_BDCP(Params..., UncSet)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("Polyhedron Model Uncertianty Set: ")
display(@test isapprox(y, [1.0, 1.0, 3.0, 2.0]; atol = 1e-5) && isapprox(objv, 17.0; atol = 1e-5))

y, objv = SolveTwoStageRO_ECCG(Params..., UncMod)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("JuMP Model Uncertianty Set: ")
display(@test (y, objv) == ([1.0, 1.0, 3.0, 2.0], 17.0))

y, objv = SolveTwoStageRO_ECCG(Params..., UncSet)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("Polyhedron Model Uncertianty Set: ")
display(@test (y, objv) == ([1.0, 1.0, 3.0, 2.0], 17.0))
## ----------------------------------------------------------------------




## --- Case-4: Two-Stage Robust Location-Transportation Problem ---
# See: Zeng, B., & Zhao, L. (2013). Solving two-stage robust optimization problems using a column-and-constraint generation method. Operations Research Letters, 41(5), 457-461.

println("Case-4: Two-Stage Robust Location-Transportation Problem")

# Model Parameter Standardization
f = [400, 414, 326]
a = [18, 25, 20]
c = [22 33 24; 33 23 30; 20 25 27]
K = fill(800,3)
Params = TSRLTP_standardize(f, a, c, K)

# Uncertainty Set Construction
# UncMod = Model()
# @variable(UncMod, d[1:3])
# @variable(UncMod, 0 <= g[1:3] <= 1)
# @constraint(UncMod, d[1] == 206 + 40g[1])
# @constraint(UncMod, d[2] == 274 + 40g[2])
# @constraint(UncMod, d[3] == 220 + 40g[3])
# @constraint(UncMod, sum(g) <= 1.8)
# @constraint(UncMod, g[1] + g[2] <= 1.2)
# UncSet = polyhedron(UncMod, CDDLib.Library(:exact))
# ind_g = ind_of_var(UncMod, "g")
# UncSet_d = eliminate(UncSet, ind_g) # There are some bugs in package `Polyhredra.jl`.

UncMod = Model()
@variable(UncMod, d[1:3])
g1 = @expression(UncMod, (1/40) * (d[1] - 206))
g2 = @expression(UncMod, (1/40) * (d[2] - 274))
g3 = @expression(UncMod, (1/40) * (d[3] - 220))
@constraint(UncMod, 0 <= g1 <= 1 )
@constraint(UncMod, 0 <= g2 <= 1 )
@constraint(UncMod, 0 <= g3 <= 1 )
@constraint(UncMod, g1 + g2 + g3 <= 1.8 )
@constraint(UncMod, g1 + g2 <= 1.2 )
UncSet_d = polyhedron(UncMod, CDDLib.Library(:exact))
all_variables(UncMod) # To check the dimension names.


# Invokating the Algorithms
y, objv = SolveTwoStageRO_CCG(Params..., UncMod)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("JuMP Model Uncertianty Set: ")
display(@test objv ≈ 33680)

y, objv = SolveTwoStageRO_CCG(Params..., UncSet_d)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("Polyhedron Model Uncertianty Set: ")
display(@test objv ≈ 33680)

y, objv = SolveTwoStageRO_BDCP(Params..., UncMod)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("JuMP Model Uncertianty Set: ")
display(@test objv ≈ 33680)

y, objv = SolveTwoStageRO_BDCP(Params..., UncSet_d)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("Polyhedron Model Uncertianty Set: ")
display(@test objv ≈ 33679.993079577296)

y, objv = SolveTwoStageRO_ECCG(Params..., UncMod)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("JuMP Model Uncertianty Set: ")
display(@test objv ≈ 33680)

y, objv = SolveTwoStageRO_ECCG(Params..., UncSet_d)
println("The facility location variable `y` is: $(TSRLTP_recover(y)[1]).")
println("The capacity variable `z` is: $(TSRLTP_recover(y)[2]).")
print("Polyhedron Model Uncertianty Set: ")
display(@test objv ≈ 33680)
## ----------------------------------------------------------------




## --- Case-5: Small Numerical Instance with Mixed Integer Recourse ---
println("Case-5: Small Numerical Instance with Mixed Integer Recourse")
c = [1]; b = [2; 3];
A = [4;;]; d = [5];
G = [6 7]; h = [8]; E = [10;;]; M = [9;;];
Sy = [Rplus()]; Sx = [Rplus(); ZeroOne()];
UncMod = Model()
@variable(UncMod, u)
δ = @expression(UncMod, 2u - 1)
@constraint(UncMod, -1 <= δ <= 1)
UncSet = polyhedron(convexhull([0], [1]))

y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncMod)
print("JuMP Model Uncertianty Set: ")
display(@test y == [1.25] && isapprox(objv, 1.25; atol = 1e-5))

y, objv = SolveTwoStageRO_CCG(c, b, A, d, G, h, E, M, Sy, Sx, UncSet)
print("Polyhedron Model Uncertianty Set: ")
display(@test y == [1.25] && isapprox(objv, 1.25; atol = 1e-5))
## --------------------------------------------------------------------




## --- Case-6: Mini Two-Stage Robust Rostering Problem ---
# See: Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.

println("Case-6: Mini Two-Stage Robust Rostering Problem")

# Model Parameter Standardization
I = 1; J = 1;
N = 8; T = 3;
c = 5 .* ones(I, T)
f = 20 .* ones(J, T)
h = 4 .* ones(J, T)
M = 40 .* ones(T)
l = 0 .* ones(I)
u = 3 .* ones(I)
a = 0 .* ones(J)
b = 3 .* ones(J)

Params = TSRTP_standardize(I, J, N, T, c, f, h, M, l, u, a, b)

# Uncertainty Set Construction
UncMod = Model()
@variable(UncMod, 5 <= d[1:T] <= 10)
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))

# Invokating the Algorithms
y, objv = SolveTwoStageRO_CCG(Params..., UncMod)
x = TSRTP_recover(y, I, T)
println("Regular staff `i` in working shift `t`: x =")
show(stdout, "text/plain", x)
println()
print("JuMP Model Uncertianty Set: ")
display(@test objv ≈ 594.00)

y, objv = SolveTwoStageRO_CCG(Params..., UncSet)
x = TSRTP_recover(y, I, T)
println("Regular staff `i` in working shift `t`: x =")
show(stdout, "text/plain", x)
println()
print("Polyhedron Model Uncertianty Set: ")
display(@test objv ≈ 594.00)
## ---------------------------------------------------------




## --- Case-7: Small Two-Stage Robust Rostering Problem ---
# See: Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.

println("Case-7: Small Two-Stage Robust Rostering Problem")

# Model Parameter Standardization
I = 3; J = 2;
N = 8; T = 3;
c = 5 .* ones(I, T)
f = 20 .* ones(J, T)
h = 4 .* ones(J, T)
M = 40 .* ones(T)
l = 1 .* ones(I)
u = 2 .* ones(I)
a = 0 .* ones(J)
b = 2 .* ones(J)

Params = TSRTP_standardize(I, J, N, T, c, f, h, M, l, u, a, b)

# Uncertainty Set Construction
UncMod = Model()
@variable(UncMod, 10 <= d[1:T] <= 15)
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))

# Invokating the Algorithms
y, objv = SolveTwoStageRO_CCG(Params..., UncMod)
x = TSRTP_recover(y, I, T)
println("Regular staff `i` in working shift `t`: x =")
show(stdout, "text/plain", x)
println()
print("JuMP Model Uncertianty Set: ")
display(@test TSRTP_recover(y, I, T) == sparse([1, 2, 3], [2, 2, 2], [1.0, 1.0, 1.0], 3, 3))

y, objv = SolveTwoStageRO_CCG(Params..., UncSet)
x = TSRTP_recover(y, I, T)
println("Regular staff `i` in working shift `t`: x =")
show(stdout, "text/plain", x)
println()
print("Polyhedron Model Uncertianty Set: ")
display(@test TSRTP_recover(y, I, T) == sparse([1, 2, 3], [2, 2, 2], [1.0, 1.0, 1.0], 3, 3))
## ---------------------------------------------------------




## --- Case-8: Two-Stage Robust Rostering Problem ---
# See: Zhao, L., & Zeng, B. (2012). An exact algorithm for two-stage robust optimization with mixed integer recourse problems. submitted, available on Optimization-Online. org.

println("Case-8: Two-Stage Robust Rostering Problem")

# Model Parameter Standardization
I = 12; J = 3;
N = 8; T = 21;
c = 5 .+ (15 - 5) .* rand(I, T)
f = 20 .+ (30 - 20) .* rand(J, T)
h = 4 .+ (8 - 4) .* rand(J, T)
M = 40 .+ (50 - 40) .* rand(T)
l = 4 .+ (8 - 4) .* rand(I)
u = 8 .+ (14 - 8) .* rand(I)
a = 2 .+ (4 - 2) .* rand(J)
b = 4 .+ (6 - 4) .* rand(J)

Params = TSRTP_standardize(I, J, N, T, c, f, h, M, l, u, a, b)

# Uncertainty Set Construction
d̲ = 30 .+ (80 - 30) .* rand(T)
ξ = 0.05 * d̲
T1 = 10 # Not mentioned in the paper
ρ1 = 0.2
ρ2 = 0.3
UncMod = Model()
@variable(UncMod, u[1:T])
@variable(UncMod, 0 <= g[1:T] <= 1)
@constraint(UncMod, u == d̲ + ξ .* g)
@constraint(UncMod, sum(g[t] for t = 1:T1+2) <= ρ1)
@constraint(UncMod, sum(g[t] for t = T1:T) <= ρ2)
UncSet = polyhedron(UncMod, CDDLib.Library(:exact))
v = all_variables(UncMod)
ind_g = ind_of_var(UncMod, "g") # `eliminate` has a bug so that we can't use `ind_of_var(UncMod, "g")`.`
UncSet_u = eliminate(UncSet, ind_g)

# Invokating the Algorithms
y, objv = SolveTwoStageRO_CCG(Params..., UncMod)
x = TSRTP_recover(y, I, T)
println("Regular staff `i` in working shift `t`: x =")
show(stdout, "text/plain", x)
println()
print("JuMP Model Uncertianty Set: ")
display(@test ~isempty(y))

y, objv = SolveTwoStageRO_CCG(Params..., UncSet_u)
x = TSRTP_recover(y, I, T)
println("Regular staff `i` in working shift `t`: x =")
show(stdout, "text/plain", x)
println()
print("Polyhedron Model Uncertianty Set: ")
display(@test ~isempty(y))
## --------------------------------------------------



############################################################################
#                                 UI Test                                  #
############################################################################

## --- Case-9: UI Test ---
println("Case-9: UI Test")
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
display(@test isapprox(y_star, [0.0]; atol = 1e-5) && isapprox(objv, 1.0; atol = 1e-5) && x_star == [0.5] && RPobjv == 0.5)

## -----------------------------


## --- Case-10: UI Test ---
println("Case-10: UI Test")
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
display(@test y_star == [1.5, 0] && objv == 54 && x_star == [6.0] && RPobjv == 18.0)
## -----------------------------


## --- Case-11: UI Test ---
println("Case-11: UI Test")

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
## -----------------------------





############################################################################
#                              MKL-aided CCG                               #
############################################################################
## --- Case-12: Test for MKL-aided CCG ---
println("Case-12: Test for MKL-aided CCG")

# Uncertain Data Generation
u1 = 0.5 .+ 4 * rand(300)
u2 = 2 ./ u1 + 0.3 * randn(300) .+ 1
X = [u1'; u2']

# MKL-based Uncertainty Set Construction
algor = HessianMKL(verbose=false)
U = mklocsvmtrain(X, 21; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05)
# mklocsvmplot(U; backend=GLMakie) # visualize the MKL uncertainty set

# Instantiate the Two-Stage RO Model 
# A two-stage robust location-transportation problem with uncertain demand data
f = [400, 414, 326]
a = [18, 25, 20]
c = [22 33;
     33 23; 
     20 25]
K = fill(5,3)
Params = TSRLTP_standardize(f, a, c, K)
model = TSROModel(Params..., U)

# Solve the Model Using MKLCCG Algorithm
mklccg = MKLCCG()
x_star, objv, recourse = solve(model; algorithm=mklccg)

# Retrieve the Recourse Decision After the Uncertainty Reveals
û = [2.0, 3.0]
y_star, RPobjv = recourse(û)
## -----------------------------------------


## --- Case-13: Test for MKL-aided CCG ---
println("Case-13: Test for MKL-aided CCG")

# Uncertain Data Generation
u1 = 0.5 .+ 4 * rand(300)
u2 = 2 ./ u1 + 0.3 * randn(300) .+ 1
X = [u1'; u2']

# MKL-based Uncertainty Set Construction
algor = HessianMKL(verbose=false)
U = mklocsvmtrain(X, 21; kernel="DPDK", algorithm=algor, q_mode="randomly", ν=0.01, μ=0.05, num_batch=3)
# mklocsvmplot(U; backend=GLMakie) # visualize the MKL uncertainty set

# Instantiate the Two-Stage RO Model 
# A two-stage robust location-transportation problem with uncertain demand data
f = [400, 414, 326]
a = [18, 25, 20]
c = [22 33;
     33 23; 
     20 25]
K = fill(5,3)
Params = TSRLTP_standardize(f, a, c, K)
model = TSROModel(Params..., U)

# Solve the Model Using MKLCCG Algorithm
mklccg = MKLCCG()
x_star, objv, recourse = solve(model; algorithm=mklccg)

# Retrieve the Recourse Decision After the Uncertainty Reveals
û = [2.0, 3.0]
y_star, RPobjv = recourse(û)
## -----------------------------------------
