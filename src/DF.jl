module DF

using TensorOperations
using JuES
using JuES.Wavefunction
using JuES.DiskTensors

export setup_df

function setup_df(refWfn::Wfn; dfbname="default")
    T = eltype(refWfn.uvsr)
    bname = refWfn.basis.blend()
    if dfbname == "default"
        if bname == "STO-3G"
            dfbname = "def2-svp-jkfit"
        else
            dfbname = "$bname-jkfit"
        end
    end
    df = psi4.core.BasisSet.build(refWfn.basis.molecule(),"DF_BASIS_MP2",dfbname)
    null = psi4.core.BasisSet.zero_ao_basis_set()
    pqP = convert(Array{T},refWfn.mints.ao_eri(refWfn.basis,refWfn.basis,df,null).np)
    Jpq = convert(Array{T},refWfn.mints.ao_eri(df,null,df,null).np)
    Jpq = squeeze(Jpq)
    pqP = squeeze(pqP)
    Jpqh = Jpq^(-1/2)
    return pqP, Jpqh
end
function make_bμν(refWfn::Wfn; dfbname=nothing)
    pqP, Jpqh = setup_df(refWfn, dfbname=dfbname)
    @tensor begin
        bμν[p,q,Q] := pqP[p,q,P]*Jpqh[P,Q]
    end
    return bμν
end

end #module
