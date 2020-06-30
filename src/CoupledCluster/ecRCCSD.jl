module ecRCCSD
using TensorOperations
using LinearAlgebra
using JuES
using JuES.Output
using JuES.Wavefunction
using JuES.IntegralTransformation
using JuES.ConfigurationInteraction.FCI
using JuES.ConfigurationInteraction.DetOperations
using JuES.CoupledCluster.AutoRCCSD

export do_ecrccsd

function cas_decomposition(Ccas::Array{Float64,1}, dets::Array{Determinant,1}, ndocc::Int, nvir::Int, frozen::Int, active::Int)

    # Initialize arrays
    T1 = zeros(ndocc, nvir)
    T2 = zeros(ndocc, ndocc, nvir, nvir)
    ecT1 = zeros(ndocc, nvir)
    ecT2 = zeros(ndocc, ndocc, nvir, nvir)

    # Build T1 and T2 from CAS
    ref = dets[1]
    for id in eachindex(dets)
       
        @inbounds D = dets[id]
        αexc = αexcitation_level(ref, D)
        βexc = βexcitation_level(ref, D)

        if αexc == 1
            
            if βexc == 0

                i, = αexclusive(ref, D)
                a = αexclusive(D, ref)[1] - ndocc

                p = phase(ref, D)

                @inbounds T1[i,a] = Ccas[id]*p

            elseif βexc == 1

                i, = αexclusive(ref, D)
                j, = βexclusive(ref, D)
                a = αexclusive(D, ref)[1] - ndocc
                b = βexclusive(D, ref)[1] - ndocc

                p = phase(ref, D)

                @inbounds T2[i,j,a,b] = Ccas[id]*p
            end

        end

    end

    @tensor T2[i,j,a,b] -= T1[i,a]*T1[j,b] 

    # Create a list of active occupied and active virtual
    actocc = []
    actvir = []

    for no in 1:(ndocc+nvir)
        
        if frozen < no ≤ ndocc
            push!(actocc, no)

        elseif ndocc < no ≤ (frozen+active)
            push!(actvir, no)

        end

    end

    println("\nActive Occ: $actocc")
    println("Active Vir: $actvir\n")



    return T1, T2, ecT1, ecT2
end

function do_ecrccsd(wfn::Wfn; kwargs...)

    # Print intro
    JuES.CoupledCluster.print_header()
    @output "\n    • Computing Externally Corrected CCSD with the ecRCCSD module.\n\n"
    
    # Process options
    for arg in keys(JuES.CoupledCluster.defaults)
        if arg in keys(kwargs)
            @eval $arg = $(kwargs[arg])
        else
            @eval $arg = $(JuES.CoupledCluster.defaults[arg])
        end
    end

    # Check if the number of electrons is even
    nelec = wfn.nalpha + wfn.nbeta
    nelec % 2 == 0 ? nothing : error("Number of electrons must be even for RHF. Given $nelec")
    nmo = wfn.nmo
    ndocc = Int(nelec/2)
    nvir = nmo - ndocc
    
    @output "\n   Calling FCI module...\n\n"
    # Compute CASCI
    λ, ϕ, dets = do_fci(wfn; kwargs...)

    Ecas = λ[1]
    Ccas = ϕ[:,1]

    # Intermediate Normalization
    abs(Ccas[1]) > 1e-8 ? nothing : error("Reference coefficient is too small ($(Ccas[1])) to performe intermediate normalization")
    Ccas = Ccas ./ Ccas[1]

    # Slices
    o = 1+fcn:ndocc
    v = ndocc+1:nmo

    # Get fock matrix
    f = get_fock(wfn; spin="alpha")

    # Save diagonal terms
    fock_Od = diag(f)[o]
    fock_Vd = diag(f)[v]
    fd = (fock_Od, fock_Vd)

    # Erase diagonal elements from original matrix
    f = f - Diagonal(f)

    # Save useful slices
    fock_OO = f[o,o]
    fock_VV = f[v,v]
    fock_OV = f[o,v]
    f = (fock_OO, fock_OV, fock_VV)

    # Get Necessary ERIs
    V = (get_eri(wfn, "OOOO", fcn=fcn), get_eri(wfn, "OOOV", fcn=fcn), get_eri(wfn, "OOVV", fcn=fcn), 
         get_eri(wfn, "OVOV", fcn=fcn), get_eri(wfn, "OVVV", fcn=fcn), get_eri(wfn, "VVVV", fcn=fcn))
    
    # Auxiliar D matrix
    fock_Od, fock_Vd = fd
    d = [i - a for i = fock_Od, a = fock_Vd]
    
    D = [i + j - a - b for i = fock_Od, j = fock_Od, a = fock_Vd, b = fock_Vd]


    newT1, newT2, ecT1, ecT2 = cas_decomposition(Ccas, dets, ndocc, nvir, frozen, active)

    # Energy from CAS vector
    Ecc = update_energy(newT1, newT2, f[2], V[3])

    @output "Energy from the CAS Vector:   {:15.10f}\n\n" Ecc+wfn.energy

    # Start CC iterations

    @output "    Starting CC Iterations\n\n"
    @output "Iteration Options:\n"
    @output "   cc_max_iter →  {:3.0d}\n" Int(cc_max_iter)
    @output "   cc_e_conv   →  {:2.0e}\n" cc_e_conv
    @output "   cc_max_rms  →  {:2.0e}\n\n" cc_max_rms
    @output "{:10s}    {: 15s}    {: 12s}    {:12s}    {:10s}\n" "Iteration" "CC Energy" "ΔE" "Max RMS" "Time (s)"

    r1 = 1
    r2 = 1
    dE = 1
    rms = 1
    ite = 1
    T1 = similar(newT1)
    T2 = similar(newT2)

    while abs(dE) > cc_e_conv || rms > cc_max_rms
        if ite > cc_max_iter
            @output "\n⚠️  CC Equations did not converge in {:1.0d} iterations.\n" cc_max_iter
            break
        end
        t = @elapsed begin

            T1 .= newT1
            T2 .= newT2
            update_amp(T1, T2, newT1, newT2, f, V)

            # Apply external correction
            newT1 += ecT1
            newT2 += ecT2

            # Apply resolvent
            newT1 ./= d
            newT2 ./= D

            # Compute residues 
            r1 = sqrt(sum((newT1 - T1).^2))/length(T1)
            r2 = sqrt(sum((newT2 - T2).^2))/length(T2)
        end
        rms = max(r1,r2)
        oldE = Ecc
        Ecc = update_energy(newT1, newT2, f[2], V[3])
        dE = Ecc - oldE
        @output "    {:<5.0d}    {:<15.10f}    {:<12.10f}    {:<12.10f}    {:<10.5f}\n" ite Ecc dE rms t
        ite += 1
    end

    # Converged?
    if abs(dE) < cc_e_conv && rms < cc_max_rms
        @output "\n 🍾 Equations Converged!\n"
    end
    @output "\n⇒ Final ecCCSD Energy:     {:15.10f}\n" Ecc+wfn.energy

    if do_pT
        Vvvvo = permutedims(V[5], [4,2,3,1])
        Vvooo = permutedims(V[2], [4,2,1,3])
        Vvovo = permutedims(V[3], [3,1,4,2])
        Ept = compute_pT(T1=T1, T2=T2, Vvvvo=Vvvvo, Vvooo=Vvooo, Vvovo=Vvovo, fo=fock_Od, fv=fock_Vd)
        @output "\n⇒ Final CCSD(T) Energy:  {:15.10f}\n" Ecc+wfn.energy+Ept
    end


end

end #Module
