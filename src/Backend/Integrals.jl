module Integrals

using PyCall
using JuES.DiskTensors

export disk_ao

"""
    disk_ao

computes all AO basis TEI and fills a DiskFourTensor object with those
"""
function disk_ao(mints::PyObject, basis::PyObject, name::String = "default")
    integ = mints.integral()
    si = integ.shells_iterator()
    si.first()
    nao = basis.nbf()
    ao_eri = DiskFourTensor("/tmp/disk_gao.$name.jues.0", Float64, nao, nao, nao, nao, "w")
    blockfill!(ao_eri, 0.0)
    while !si.is_done()
        p, q, r, s = (si.p, si.q, si.r, si.s)
        pf = basis.shell_to_basis_function(p) + 1
        qf = basis.shell_to_basis_function(q) + 1
        rf = basis.shell_to_basis_function(r) + 1
        sf = basis.shell_to_basis_function(s) + 1
        shell = mints.ao_eri_shell(p, q, r, s).to_array()
        pn, qn, rn, sn = size(shell)
        pn -= 1
        qn -= 1
        rn -= 1
        sn -= 1
        ao_eri[pf:pf+pn, qf:qf+qn, rf:rf+rn, sf:sf+sn] = shell
        ao_eri[qf:qf+qn, pf:pf+pn, rf:rf+rn, sf:sf+sn] = permutedims(shell, [2, 1, 3, 4])
        ao_eri[pf:pf+pn, qf:qf+qn, sf:sf+sn, rf:rf+rn] = permutedims(shell, [1, 2, 4, 3])
        ao_eri[qf:qf+qn, pf:pf+pn, sf:sf+sn, rf:rf+rn] =
            permutedims(permutedims(shell, [2, 1, 3, 4]), [1, 2, 4, 3])
        ao_eri[rf:rf+rn, sf:sf+sn, pf:pf+pn, qf:qf+qn] =
            permutedims(permutedims(shell, [3, 2, 1, 4]), [1, 4, 3, 2])
        ao_eri[sf:sf+sn, rf:rf+rn, pf:pf+pn, qf:qf+qn] = permutedims(
            permutedims(permutedims(shell, [3, 2, 1, 4]), [1, 4, 3, 2]),
            [2, 1, 3, 4],
        )
        ao_eri[rf:rf+rn, sf:sf+sn, qf:qf+qn, pf:pf+pn] = permutedims(
            permutedims(permutedims(shell, [3, 2, 1, 4]), [1, 4, 3, 2]),
            [1, 2, 4, 3],
        )
        ao_eri[sf:sf+sn, rf:rf+rn, qf:qf+qn, pf:pf+pn] =
            permutedims(permutedims(shell, [4, 2, 3, 1]), [1, 3, 2, 4])
        si.next()
    end
    return ao_eri

end

"""
    get_eri 

return a specified ERI array.

Arguments:

wfn        -> Wavefunctions object
eri_string -> String with length 4 identifying the type of ERI. 
              Characters must be (o, O, v, V). Each indicating Occupied and Virtual for ALPHA and beta.

notation   -> OPTIONAL. Values: "chem" or "phys". Retuning the array in Chemist's or Physicists' notation.

"""

function get_eri(wfn::Wfn, eri_string::String, notation::String = "phys")

    # Get C1, C2, C3, C4 for the integral transformation

end

end #module
