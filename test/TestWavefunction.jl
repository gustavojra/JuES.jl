using Test
using PyCall
using JuES.Wavefunction
psi4 = pyimport("psi4")
psi4.core.be_quiet() #turn off output
# > setup
tol = 1E-14
mol = psi4.geometry("""
					H
					H 1 1.0
					symmetry c1
					""")
psi4.set_options(Dict("scf_type" => "pk","d_convergence"=>14))
e,wfn = psi4.energy("hf/sto-3g",mol=mol,return_wfn=true)
JuWfn = PyToJl(wfn,Float64,false)
@testset "Wavefunction" begin
	@testset "Smoke" begin
		@testset "Attributes" begin
			@test JuWfn.nalpha == 1 && JuWfn.nbeta == 1
		end
		@testset "Integrals" begin
			@test (JuWfn.uvsr[1,1,1,1] - 0.7746059439198979) < tol
			@test (JuWfn.uvsr[2,1,2,2] - 0.3093089669634818) < tol
			@test (JuWfn.uvsr[1,1,2,2] - 0.4780413730018048) < tol
		end
	end
end
