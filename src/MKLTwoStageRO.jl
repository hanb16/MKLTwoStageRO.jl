module MKLTwoStageRO

using Reexport
using JuMP: @constraint, @expression, @objective, @variable, ALMOST_LOCALLY_SOLVED, ITERATION_LIMIT, LOCALLY_SOLVED, Model, NORM_LIMIT, OPTIMAL, all_variables, delete, dual, name, num_variables, object_dictionary, objective_value, optimize!, set_normalized_rhs, set_optimizer, set_optimizer_attributes, set_silent, termination_status, unregister, value, NUMERICAL_ERROR
@reexport using Polyhedra
@reexport using CDDLib
using HiGHS
using Ipopt
using Printf: @printf, @sprintf, Format, format
@reexport using MKLOneClassSVM

export @constraint, @expression, @variable, Model, all_variables # From JuMP




# Supported Basic Domains
include("types.jl")
export Rplus, ZeroOne, Zplus
export TSROModel
export CCG, BDCP, ECCG, MKLCCG

# The (Nested) Column-and-Constraint Generation Algorithm
include("column_and_constraint_generation_method.jl")
export SolveTwoStageRO_CCG

# The Benders-Dual Cutting Plane Method
include("benders_dual_cutting_plane_method.jl")
export SolveTwoStageRO_BDCP

# The Extended Column-and-Constraint Generation Algorithm
include("extended_column_and_constraint_generation_method.jl")
export SolveTwoStageRO_ECCG

# The Multiple Kernel Learning aided C&CG Algorithm
include("multiple_kernel_learning_aided_ccg.jl")
export SolveTwoStageRO_MKLCCG

# Utilities
include("utilities.jl")
export ind_of_var
export solve


end # module MKLTwoStageRO
