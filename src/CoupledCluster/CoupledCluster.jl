"""
Basic module for running CC computations in Julia.

short term goal is RCCD and UCCD
medium term goal is RCCSD and UCCSD
long term goal is RCCSD(T) and UCCSD(T)

Implemented --> RCCD, RCCSD, DF-RCCD


usage --> methods should be defined like do_<r/u><method> and take in
	  --> a Wavefunction.jl Wfn object as their sole _required_ input.
	  --> optional inputs such as maxit, convergence, etc can be defined
	  --> via multiple dispatch
"""
module CoupledCluster

using JuES.Wavefunction
using JuES.Transformation
using Base.Threads
#using SharedArrays
#using Distributed
using TensorOperations
using LinearAlgebra
using Dates

include("RCCD.jl")
include("DF-RCCD.jl")
include("ROCCD.jl")
include("RCCSD.jl")
include("UCCSD.jl")
include("unf-RCCSD.jl")
end #module CC