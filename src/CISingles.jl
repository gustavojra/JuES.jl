module CISingles
"""
Module for performing CIS computations.

Currently only RHF reference supported
"""

using IterativeSolvers
using JuES.Wavefunction
using JuES.Transformation
using PyCall
using LinearAlgebra 
using Dates

const psi4 = PyNULL()

function __init__()
	copy!(psi4,pyimport("psi4"))
end
dt = Float64

#export setup_rcis
export do_RCIS

#function setup_rcis(wfn,dt)
#    _C = wfn.Ca()
#    nbf = wfn.nmo()
#    #nso = 2*nbf
#    #nocc = 2*wfn.nalpha()
#    nso = nbf
#    nocc = wfn.nalpha()
#    eps = wfn.epsilon_a().to_array()
#    nvir = nso - nocc
#    basis = wfn.basisset()
#    mints = psi4.core.MintsHelper(basis)
#    mo_eri = convert(Array{dt},mints.mo_eri(_C,_C,_C,_C).to_array())
#    ff = zeros(dt,nso,nso)
#    r = collect(UnitRange(1,nso))
#    @inbounds @fastmath for i in r
#        ff[i,i] = eps[Int64(fld((i+1),2))]
#    end
#    return nocc,nvir,mo_eri,ff
#end


@fastmath function do_RCIS(refWfn::Wfn,nroots,algo="lobpcg",doprint=false)
	nocc = 2*refWfn.nalpha
	nvir = 2*refWfn.nvira
    eri_voov = tei_transform(refWfn.uvsr,refWfn.Cav,refWfn.Cao,refWfn.Cao,refWfn.Cav,"eri_voov")
    eri_vvoo = tei_transform(refWfn.uvsr,refWfn.Cav,refWfn.Cav,refWfn.Cao,refWfn.Cao,"eri_vvoo")
	F = zeros(nocc + nvir, nocc + nvir)
	r = collect(UnitRange(1,nocc+nvir))
    @inbounds @fastmath for i in r
        F[i,i] = refWfn.epsa[Int64(fld((i+1),2))]
    end
    if doprint println("constructing Hamiltonian ") end
    dt = @elapsed H = make_H(nocc,nvir,eri_voov,eri_vvoo,F)
    if doprint println("Hamiltonian constructed in $dt s") end

	#branch for eigenvalue algorithm
    if algo == "lobpcg" || algo == "iter"
        dt = @elapsed eigs = lobpcg(H,false,nroots).λ
        if doprint println("Hamiltonian iteratively solved in $dt s") end
	elseif algo == "davidson"
		println(size(H))
        dt = @elapsed eigs = eigdav(H,1,4,100,1E-6)
		if doprint println("Hamiltonian iteratively solved in $dt s") end
	elseif algo == "svd"
		dt = @elapsed eigs = svdvals(H)
        if doprint println("Hamiltonian diagonalized exactly in $dt s") end
    elseif algo == "diag"
    	H = Symmetric(H)
        dt = @elapsed eigs = eigvals(H)#,1:nroots)
        if doprint println("Hamiltonian diagonalized exactly in $dt s") end
    else
        if doprint
			println("solver ",algo," is not supported! Choose from { lobpcg (synonym iter), davidson, svd, diag }")
        end
        return false
    end
    return eigs#[1:nroots]
end

"""
    make_H

constructs Hamiltonian matrix for CI singles.
"""
function make_H(nocc,nvir,eri_voov,eri_vvoo,F)
    H = zeros(dt,nocc*nvir,nocc*nvir)
    rocc = collect(UnitRange(1,nocc))
    rvir = collect(UnitRange(1,nvir))
    for b in rvir
        for i in rocc
            for j in rocc
                for a in rvir
                    #aa = a + nocc
                    #bb = b + nocc
                    I = (i-1)*nvir+a
                    J = (j-1)*nvir+b
                    H[I,J] = HS(eri_voov,eri_vvoo,F,i,j,a,b)
                end
            end
        end
    end
    return H
end

"""
    HS

computes a singly excited Hamiltonian matrix element. Slaters rules are
incorporated via the kronecker deltas
"""
function HS(eri_voov::Array{dt,4},eri_vvoo::Array{dt,4},
                                        F::Array{dt,2}, 
                                        i::Int64, j::Int64, a::Int64, b::Int64)
    return kron(i,j)*F[a,b] - kron(a,b)*F[i,j] #+ so_eri(eri_voov,eri_vvoo,a,j,i,b)
end

function so_eri(eri_voov,eri_vvoo,a,j,i,b)
    aa = Int64(fld((a+1),2))
    jj = Int64(fld((j+1),2))
    ii = Int64(fld((i+1),2))
    bb = Int64(fld((b+1),2))
    #<aj||ib> = 
    (a%2==i%2)*(j%2==b%2)*eri_voov[aa,ii,jj,bb] - (a%2==b%2)*(j%2==i%2)*eri_vvoo[aa,bb,jj,ii]
end


@inline @fastmath function kron(a::Int64,b::Int64)
    "Kronecker delta function"
    @fastmath return a == b
end
end #module
