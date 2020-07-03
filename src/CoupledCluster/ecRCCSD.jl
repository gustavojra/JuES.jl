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

function print_nicer(M)

    println("\n")
    for i in 1:size(M)[1]
        println(M[i,:])
    end
    println("\n")
end

function cas_decomposition(Ccas::Array{Float64,1}, dets::Array{Determinant,1}, ndocc::Int, nvir::Int, frozen::Int, active::Int, f::Tuple, V::Tuple, fcn::Int)

    Voooo, Vooov, Voovv, Vovov, Vovvv, Vvvvv = V
    fock_OO, fock_OV, fock_VV = f

    # Initialize arrays
    T1 = zeros(size(fock_OV))
    T2 = zeros(size(Voovv))
    ecT1 = zeros(size(fock_OV))
    ecT2 = zeros(size(Voovv))

    # Build T1 and T2 from CAS
    ref = dets[1]
    for id in eachindex(dets)
       
        @inbounds D = dets[id]
        αexc = αexcitation_level(ref, D)
        βexc = βexcitation_level(ref, D)

        if αexc == 1
            
            if βexc == 0

                i, = αexclusive(ref, D) .- fcn
                a, = αexclusive(D, ref) .- ndocc

                p = phase(ref, D)

                @inbounds T1[i,a] = Ccas[id]*p

            elseif βexc == 1

                i, = αexclusive(ref, D) .- fcn
                j, = βexclusive(ref, D) .- fcn
                a, = αexclusive(D, ref) .- ndocc
                b, = βexclusive(D, ref) .- ndocc

                p = phase(ref, D)

                @inbounds T2[i,j,a,b] = Ccas[id]*p
            end

        elseif (αexc + βexc) > 2
            # This line relies on the fact the dets are ordered by excitation level
            break
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

    # Build T3, T4 and T4aa
    T3 = zeros(ndocc-fcn, ndocc-fcn, ndocc-fcn, nvir, nvir, nvir)
    T4 = zeros(ndocc-fcn, ndocc-fcn, ndocc-fcn, ndocc-fcn, nvir, nvir, nvir, nvir)
    T4aa = zeros(ndocc-fcn, ndocc-fcn, ndocc-fcn, ndocc-fcn, nvir, nvir, nvir, nvir)

    for id in eachindex(dets)
       
        @inbounds D = dets[id]
        αexc = αexcitation_level(ref, D)
        βexc = βexcitation_level(ref, D)

        if αexc == 2 && βexc == 1
            
            # i > k, a > c
            k,i = αexclusive(ref, D) .- fcn
            j,  = βexclusive(ref, D) .- fcn
            c,a = αexclusive(D,ref) .- ndocc
            b,  = βexclusive(D, ref) .- ndocc

            p = phase(ref, D)

            T3[i,j,k,a,b,c] =  p*Ccas[id]
            T3[k,j,i,a,b,c] = -p*Ccas[id]
            T3[i,j,k,c,b,a] = -p*Ccas[id]
            T3[k,j,i,c,b,a] =  p*Ccas[id]


        elseif αexc == 3 && βexc == 1
            
            # i > k > l, a > c > d
            l,k,i = αexclusive(ref, D) .- fcn
            j, = βexclusive(ref, D) .- fcn
            d,c,a = αexclusive(D,ref) .- ndocc
            b, = βexclusive(D, ref) .- ndocc

            p = phase(ref, D)

            T4aa[i,j,k,l,a,b,c,d] =  p*Ccas[id]
            T4aa[i,j,k,l,c,b,d,a] =  p*Ccas[id]
            T4aa[i,j,k,l,a,b,d,c] = -p*Ccas[id]
            T4aa[i,j,k,l,c,b,a,d] = -p*Ccas[id]
            T4aa[i,j,k,l,d,b,a,c] =  p*Ccas[id]
            T4aa[i,j,k,l,d,b,c,a] = -p*Ccas[id]
            T4aa[k,j,i,l,c,b,d,a] = -p*Ccas[id]
            T4aa[k,j,i,l,a,b,d,c] =  p*Ccas[id]
            T4aa[k,j,i,l,c,b,a,d] =  p*Ccas[id]
            T4aa[k,j,i,l,a,b,c,d] = -p*Ccas[id]
            T4aa[k,j,i,l,d,b,a,c] = -p*Ccas[id]
            T4aa[k,j,i,l,d,b,c,a] =  p*Ccas[id]
            T4aa[k,j,l,i,c,b,d,a] =  p*Ccas[id]
            T4aa[k,j,l,i,a,b,d,c] = -p*Ccas[id]
            T4aa[k,j,l,i,c,b,a,d] = -p*Ccas[id]
            T4aa[k,j,l,i,a,b,c,d] =  p*Ccas[id]
            T4aa[k,j,l,i,d,b,a,c] =  p*Ccas[id]
            T4aa[k,j,l,i,d,b,c,a] = -p*Ccas[id]
            T4aa[l,j,k,i,c,b,d,a] = -p*Ccas[id]
            T4aa[l,j,k,i,a,b,d,c] =  p*Ccas[id]
            T4aa[l,j,k,i,c,b,a,d] =  p*Ccas[id]
            T4aa[l,j,k,i,a,b,c,d] = -p*Ccas[id]
            T4aa[l,j,k,i,d,b,a,c] = -p*Ccas[id]
            T4aa[l,j,k,i,d,b,c,a] =  p*Ccas[id]
            T4aa[l,j,i,k,c,b,d,a] =  p*Ccas[id]
            T4aa[l,j,i,k,a,b,d,c] = -p*Ccas[id]
            T4aa[l,j,i,k,c,b,a,d] = -p*Ccas[id]
            T4aa[l,j,i,k,a,b,c,d] =  p*Ccas[id]
            T4aa[l,j,i,k,d,b,a,c] =  p*Ccas[id]
            T4aa[l,j,i,k,d,b,c,a] = -p*Ccas[id]
            T4aa[i,j,l,k,c,b,d,a] = -p*Ccas[id]
            T4aa[i,j,l,k,a,b,d,c] =  p*Ccas[id]
            T4aa[i,j,l,k,c,b,a,d] =  p*Ccas[id]
            T4aa[i,j,l,k,a,b,c,d] = -p*Ccas[id]
            T4aa[i,j,l,k,d,b,a,c] = -p*Ccas[id]
            T4aa[i,j,l,k,d,b,c,a] =  p*Ccas[id]


        elseif αexc == 2 && βexc == 2
            
            # i > k, j > l, a > c, b > d
            k,i = αexclusive(ref, D) .- fcn
            l,j = βexclusive(ref, D) .- fcn
            c,a = αexclusive(D,ref) .- ndocc
            d,b = βexclusive(D, ref) .- ndocc

            p = phase(ref, D)

            T4[i,j,k,l,a,b,c,d] =  p*Ccas[id]
            T4[i,j,k,l,a,d,c,b] = -p*Ccas[id]
            T4[i,j,k,l,c,d,a,b] =  p*Ccas[id]
            T4[i,j,k,l,c,b,a,d] = -p*Ccas[id]
            T4[k,j,i,l,a,d,c,b] =  p*Ccas[id]
            T4[k,j,i,l,a,b,c,d] = -p*Ccas[id]
            T4[k,j,i,l,c,d,a,b] = -p*Ccas[id]
            T4[k,j,i,l,c,b,a,d] =  p*Ccas[id]
            T4[k,l,i,j,a,d,c,b] = -p*Ccas[id]
            T4[k,l,i,j,a,b,c,d] =  p*Ccas[id]
            T4[k,l,i,j,c,d,a,b] =  p*Ccas[id]
            T4[k,l,i,j,c,b,a,d] = -p*Ccas[id]
            T4[i,l,k,j,a,d,c,b] =  p*Ccas[id]
            T4[i,l,k,j,a,b,c,d] = -p*Ccas[id]
            T4[i,l,k,j,c,d,a,b] = -p*Ccas[id]
            T4[i,l,k,j,c,b,a,d] =  p*Ccas[id]

        end

    end

    # Cluster Decomposition of T3
    @tensoropt begin
        T3[i,j,k,a,b,c] -= T1[i,a]*T1[j,b]*T1[k,c]
        T3[i,j,k,a,b,c] += T1[i,c]*T1[j,b]*T1[k,a]
        T3[i,j,k,a,b,c] -= T1[i,a]*T2[k,j,c,b]
        T3[i,j,k,a,b,c] += T1[k,a]*T2[i,j,c,b]
        T3[i,j,k,a,b,c] += T1[i,c]*T2[k,j,a,b]
        T3[i,j,k,a,b,c] -= T1[k,c]*T2[i,j,a,b]
        T3[i,j,k,a,b,c] += T1[j,b]*T2[k,i,a,c]
        T3[i,j,k,a,b,c] -= T1[j,b]*T2[i,k,a,c]
    end

    # Cluster Decomposition of T4aa
    @tensoropt begin
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[i,a]*T1[k,c]*T1[l,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T1[i,a]*T1[l,c]*T1[k,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T1[i,a]*T2[l,k,c,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[i,a]*T2[k,l,c,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[k,c]*T1[i,a]*T2[l,j,d,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[l,c]*T1[i,a]*T2[k,j,d,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,d]*T1[i,a]*T2[l,j,c,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,d]*T1[i,a]*T2[k,j,c,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,a]*T3[k,j,l,c,b,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T1[k,a]*T1[i,c]*T1[l,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[k,a]*T1[l,c]*T1[i,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[k,a]*T2[l,i,c,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T1[k,a]*T2[i,l,c,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[i,c]*T1[k,a]*T2[l,j,d,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,c]*T1[k,a]*T2[i,j,d,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,d]*T1[k,a]*T2[l,j,c,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[l,d]*T1[k,a]*T2[i,j,c,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,a]*T3[i,j,l,c,b,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[l,a]*T1[i,c]*T1[k,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T1[l,a]*T1[k,c]*T1[i,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T1[l,a]*T2[k,i,c,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[l,a]*T2[i,k,c,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,c]*T1[l,a]*T2[k,j,d,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,c]*T1[l,a]*T2[i,j,d,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[i,d]*T1[l,a]*T2[k,j,c,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[k,d]*T1[l,a]*T2[i,j,c,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,a]*T3[i,j,k,c,b,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,c]*T1[j,b]*T2[l,k,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[i,c]*T1[j,b]*T2[k,l,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,c]*T1[j,b]*T2[l,i,a,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[k,c]*T1[j,b]*T2[i,l,a,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,c]*T1[j,b]*T2[k,i,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[l,c]*T1[j,b]*T2[i,k,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[i,d]*T1[j,b]*T2[l,k,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,d]*T1[j,b]*T2[k,l,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T1[k,d]*T1[j,b]*T2[l,i,a,c]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,d]*T1[j,b]*T2[i,l,a,c]
        T4aa[i,j,k,l,a,b,c,d] += T1[l,d]*T1[j,b]*T2[k,i,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,d]*T1[j,b]*T2[i,k,a,c]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T3[k,i,l,a,c,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[j,b]*T3[i,k,l,a,c,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[j,b]*T3[i,l,k,a,c,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[k,d]*T1[i,c]*T2[l,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[l,d]*T1[i,c]*T2[k,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[i,c]*T3[k,j,l,a,b,d]
        T4aa[i,j,k,l,a,b,c,d] += T1[i,d]*T1[k,c]*T2[l,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,d]*T1[k,c]*T2[i,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] -= T1[k,c]*T3[i,j,l,a,b,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,d]*T1[l,c]*T2[k,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,d]*T1[l,c]*T2[i,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T1[l,c]*T3[i,j,k,a,b,d]
        T4aa[i,j,k,l,a,b,c,d] -= T1[i,d]*T3[k,j,l,a,b,c]
        T4aa[i,j,k,l,a,b,c,d] += T1[k,d]*T3[i,j,l,a,b,c]
        T4aa[i,j,k,l,a,b,c,d] -= T1[l,d]*T3[i,j,k,a,b,c]
        T4aa[i,j,k,l,a,b,c,d] += T2[l,k,c,d]*T2[i,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] -= T2[k,l,c,d]*T2[i,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] -= T2[l,i,c,d]*T2[k,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T2[i,l,c,d]*T2[k,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T2[k,i,c,d]*T2[l,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] -= T2[i,k,c,d]*T2[l,j,a,b]
        T4aa[i,j,k,l,a,b,c,d] += T2[l,j,d,b]*T2[k,i,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T2[l,j,d,b]*T2[i,k,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T2[k,j,d,b]*T2[l,i,a,c]
        T4aa[i,j,k,l,a,b,c,d] += T2[k,j,d,b]*T2[i,l,a,c]
        T4aa[i,j,k,l,a,b,c,d] += T2[i,j,d,b]*T2[l,k,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T2[i,j,d,b]*T2[k,l,a,c]
        T4aa[i,j,k,l,a,b,c,d] -= T2[l,j,c,b]*T2[k,i,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T2[l,j,c,b]*T2[i,k,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T2[k,j,c,b]*T2[l,i,a,d]
        T4aa[i,j,k,l,a,b,c,d] -= T2[k,j,c,b]*T2[i,l,a,d]
        T4aa[i,j,k,l,a,b,c,d] -= T2[i,j,c,b]*T2[l,k,a,d]
        T4aa[i,j,k,l,a,b,c,d] += T2[i,j,c,b]*T2[k,l,a,d]
    end

    # Cluster decomposition of T4
    @tensoropt begin
        T4[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[i,a]*T1[k,c]*T1[l,d]
        T4[i,j,k,l,a,b,c,d] -= T1[j,b]*T1[i,a]*T2[k,l,c,d]
        T4[i,j,k,l,a,b,c,d] += T1[l,b]*T1[i,a]*T1[k,c]*T1[j,d]
        T4[i,j,k,l,a,b,c,d] += T1[l,b]*T1[i,a]*T2[k,j,c,d]
        T4[i,j,k,l,a,b,c,d] += T1[k,c]*T1[i,a]*T2[l,j,b,d]
        T4[i,j,k,l,a,b,c,d] -= T1[k,c]*T1[i,a]*T2[j,l,b,d]
        T4[i,j,k,l,a,b,c,d] += T1[j,d]*T1[i,a]*T2[k,l,c,b]
        T4[i,j,k,l,a,b,c,d] -= T1[l,d]*T1[i,a]*T2[k,j,c,b]
        T4[i,j,k,l,a,b,c,d] -= T1[i,a]*T3[j,k,l,b,c,d]
        T4[i,j,k,l,a,b,c,d] += T1[j,b]*T1[k,a]*T1[i,c]*T1[l,d]
        T4[i,j,k,l,a,b,c,d] += T1[j,b]*T1[k,a]*T2[i,l,c,d]
        T4[i,j,k,l,a,b,c,d] -= T1[l,b]*T1[k,a]*T1[i,c]*T1[j,d]
        T4[i,j,k,l,a,b,c,d] -= T1[l,b]*T1[k,a]*T2[i,j,c,d]
        T4[i,j,k,l,a,b,c,d] -= T1[i,c]*T1[k,a]*T2[l,j,b,d]
        T4[i,j,k,l,a,b,c,d] += T1[i,c]*T1[k,a]*T2[j,l,b,d]
        T4[i,j,k,l,a,b,c,d] -= T1[j,d]*T1[k,a]*T2[i,l,c,b]
        T4[i,j,k,l,a,b,c,d] += T1[l,d]*T1[k,a]*T2[i,j,c,b]
        T4[i,j,k,l,a,b,c,d] += T1[k,a]*T3[j,i,l,b,c,d]
        T4[i,j,k,l,a,b,c,d] += T1[i,c]*T1[j,b]*T2[k,l,a,d]
        T4[i,j,k,l,a,b,c,d] -= T1[k,c]*T1[j,b]*T2[i,l,a,d]
        T4[i,j,k,l,a,b,c,d] += T1[l,d]*T1[j,b]*T2[k,i,a,c]
        T4[i,j,k,l,a,b,c,d] -= T1[l,d]*T1[j,b]*T2[i,k,a,c]
        T4[i,j,k,l,a,b,c,d] -= T1[j,b]*T3[i,l,k,a,d,c]
        T4[i,j,k,l,a,b,c,d] -= T1[i,c]*T1[l,b]*T2[k,j,a,d]
        T4[i,j,k,l,a,b,c,d] += T1[k,c]*T1[l,b]*T2[i,j,a,d]
        T4[i,j,k,l,a,b,c,d] -= T1[j,d]*T1[l,b]*T2[k,i,a,c]
        T4[i,j,k,l,a,b,c,d] += T1[j,d]*T1[l,b]*T2[i,k,a,c]
        T4[i,j,k,l,a,b,c,d] += T1[l,b]*T3[i,j,k,a,d,c]
        T4[i,j,k,l,a,b,c,d] -= T1[j,d]*T1[i,c]*T2[k,l,a,b]
        T4[i,j,k,l,a,b,c,d] += T1[l,d]*T1[i,c]*T2[k,j,a,b]
        T4[i,j,k,l,a,b,c,d] += T1[i,c]*T3[j,k,l,b,a,d]
        T4[i,j,k,l,a,b,c,d] += T1[j,d]*T1[k,c]*T2[i,l,a,b]
        T4[i,j,k,l,a,b,c,d] -= T1[l,d]*T1[k,c]*T2[i,j,a,b]
        T4[i,j,k,l,a,b,c,d] -= T1[k,c]*T3[j,i,l,b,a,d]
        T4[i,j,k,l,a,b,c,d] += T1[j,d]*T3[i,l,k,a,b,c]
        T4[i,j,k,l,a,b,c,d] -= T1[l,d]*T3[i,j,k,a,b,c]
        T4[i,j,k,l,a,b,c,d] -= T2[k,l,c,d]*T2[i,j,a,b]
        T4[i,j,k,l,a,b,c,d] += T2[k,j,c,d]*T2[i,l,a,b]
        T4[i,j,k,l,a,b,c,d] += T2[i,l,c,d]*T2[k,j,a,b]
        T4[i,j,k,l,a,b,c,d] -= T2[i,j,c,d]*T2[k,l,a,b]
        T4[i,j,k,l,a,b,c,d] -= T2[l,j,b,d]*T2[k,i,a,c]
        T4[i,j,k,l,a,b,c,d] += T2[l,j,b,d]*T2[i,k,a,c]
        T4[i,j,k,l,a,b,c,d] += T2[j,l,b,d]*T2[k,i,a,c]
        T4[i,j,k,l,a,b,c,d] -= T2[j,l,b,d]*T2[i,k,a,c]
        T4[i,j,k,l,a,b,c,d] += T2[k,l,c,b]*T2[i,j,a,d]
        T4[i,j,k,l,a,b,c,d] -= T2[k,j,c,b]*T2[i,l,a,d]
        T4[i,j,k,l,a,b,c,d] -= T2[i,l,c,b]*T2[k,j,a,d]
        T4[i,j,k,l,a,b,c,d] += T2[i,j,c,b]*T2[k,l,a,d]
    end


    # Compute ecT1
    for n in (actocc .- fcn)
        for f in (actvir .- ndocc)
            T3_3n6f = view(T3, :, :, n, :, :, f)
            Voovv_1n4f = view(Voovv, n, :, :, f)
            Voovv_2n4f = view(Voovv, :, n, :, f)
            Voovv_1n3f = view(Voovv, n, :, f, :)
            Vovvv_1n4f = view(Vovvv, n, :, :, f)
            Vovvv_1n3f = view(Vovvv, n, :, f, :)
            Vooov_1n3f = view(Vooov, n, :, f, :)
            Vooov_1n4f = view(Vooov, n, :, :, f)
            Vooov_2n4f = view(Vooov, :, n, :, f)
            fock_OV_1n2f = fock_OV[n,f]

            @tensoropt begin

                # Compute ecT1
                ecT1[i,a] += 0.25*T3_3n6f[m,i,e,a]*Voovv_2n4f[m,e]
                ecT1[i,a] += 1.5*T3_3n6f[i,m,a,e]*Voovv_2n4f[m,e]
                ecT1[i,a] += -0.25*T3_3n6f[m,i,a,e]*Voovv_2n4f[m,e]
                ecT1[i,a] += -0.5*T3_3n6f[i,m,a,e]*Voovv_1n4f[m,e]
                ecT1[i,a] += -0.25*T3_3n6f[m,i,e,a]*Voovv_1n4f[m,e]
                ecT1[i,a] += 0.25*T3_3n6f[m,i,a,e]*Voovv_1n4f[m,e]

                # Compute ecT2
                ecT2[i,j,a,b] += fock_OV_1n2f*T3_3n6f[j,i,b,a]   # Maybe those two are the same?
                ecT2[i,j,a,b] += fock_OV_1n2f*T3_3n6f[i,j,a,b]
                ecT2[i,j,a,b] += -0.5*T3_3n6f[i,j,e,b]*Vovvv_1n4f[a,e]
                ecT2[i,j,a,b] += 0.5*T3_3n6f[i,j,e,b]*Vovvv_1n3f[a,e]
                ecT2[i,j,a,b] += T3_3n6f[j,i,b,e]*Vovvv_1n3f[a,e]
                ecT2[i,j,a,b] += 0.5*T3_3n6f[m,j,a,b]*Vooov_1n4f[m,i]
                ecT2[i,j,a,b] += -0.5*T3_3n6f[m,j,a,b]*Vooov_2n4f[m,i]
                ecT2[i,j,a,b] -= T3_3n6f[j,m,b,a]*Vooov_2n4f[m,i]
                ecT2[i,j,a,b] += 0.5*T3_3n6f[m,i,b,a]*Vooov_1n4f[m,j]
                ecT2[i,j,a,b] -= T3_3n6f[i,m,a,b]*Vooov_2n4f[m,j]
                ecT2[i,j,a,b] += -0.5*T3_3n6f[m,i,b,a]*Vooov_2n4f[m,j]
                ecT2[i,j,a,b] += -0.5*T3_3n6f[j,i,e,a]*Vovvv_1n4f[b,e]
                ecT2[i,j,a,b] += T3_3n6f[i,j,a,e]*Vovvv_1n3f[b,e]
                ecT2[i,j,a,b] += 0.5*T3_3n6f[j,i,e,a]*Vovvv_1n3f[b,e]
                ecT2[i,j,a,b] -= T1[m,b]*T3_3n6f[i,j,a,e]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] += -0.5*T1[m,b]*T3_3n6f[j,i,e,a]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] += 0.5*T1[m,b]*T3_3n6f[j,i,e,a]*Voovv_1n4f[m,e]
                ecT2[i,j,a,b] += 0.5*T1[m,a]*T3_3n6f[i,j,e,b]*Voovv_1n4f[m,e]
                ecT2[i,j,a,b] += -0.5*T1[m,a]*T3_3n6f[i,j,e,b]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] -= T1[m,a]*T3_3n6f[j,i,b,e]*Voovv_1n3f[m,e]
                ecT2[i,j,a,b] -= T1[j,e]*T3_3n6f[i,m,a,b]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] += 0.5*T1[j,e]*T3_3n6f[m,i,b,a]*Voovv_1n4f[m,e]
                ecT2[i,j,a,b] += -0.5*T1[j,e]*T3_3n6f[m,i,b,a]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] += 0.5*T1[i,e]*T3_3n6f[m,j,a,b]*Voovv_1n4f[m,e]
                ecT2[i,j,a,b] += -0.5*T1[i,e]*T3_3n6f[m,j,a,b]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] -= T1[i,e]*T3_3n6f[j,m,b,a]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] -= T1[m,e]*T3_3n6f[i,j,a,b]*Voovv_1n4f[m,e]
                ecT2[i,j,a,b] += 2.0*T1[m,e]*T3_3n6f[i,j,a,b]*Voovv_2n4f[m,e]
                ecT2[i,j,a,b] -= T1[m,e]*T3_3n6f[j,i,b,a]*Voovv_1n4f[m,e]
                ecT2[i,j,a,b] += 2.0*T1[m,e]*T3_3n6f[j,i,b,a]*Voovv_2n4f[m,e]
            end

            for m in (actocc .- fcn)
                for e in (actvir .- ndocc)

                    T4_3m4n7e8f = view(T4, :, :, m, n, :, :, e, f)
                    T4aa_3m4n7e8f = view(T4aa, :, :, m, n, :, :, e, f)
                    Voovv_1m2n3e4f = Voovv[m,n,e,f]
                    Voovv_1n2m3e4f = Voovv[n,m,e,f]

                    ecT2 += T4_3m4n7e8f.*Voovv_1m2n3e4f
                    ecT2 += 0.25.*T4aa_3m4n7e8f.*(Voovv_1m2n3e4f - Voovv_1n2m3e4f)
                    ecT2 += 0.25.*permutedims(T4aa_3m4n7e8f, [2,1,4,3]).*(Voovv_1m2n3e4f - Voovv_1n2m3e4f)
                end
            end
        end
    end

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
    vnuc = 8.888064120957244
    
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


    @output "Starting Cluster Decomposition...\n"
    @time newT1, newT2, ecT1, ecT2 = cas_decomposition(Ccas, dets, ndocc, nvir, frozen, active, f, V, fcn)

    # Energy from CAS vector
    Ecc = update_energy(newT1, newT2, f[2], V[3])

    @output "Energy from the CAS Vector:   {:15.10f}\n\n" Ecc+wfn.energy+vnuc

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
    @output "\n⇒ Final ecCCSD Energy:     {:15.10f}\n" Ecc+wfn.energy+vnuc

end

end #Module
