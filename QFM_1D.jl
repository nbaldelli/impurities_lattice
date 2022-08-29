#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, SparseArrays, Combinatorics, ITensors, ITensors.HDF5, MKL
using DelimitedFiles, LaTeXStrings

  
function ITensors.op(::OpName"Sz", ::SiteType"Boson", s1::Index)
    sz = 1/2*op("Id", s1) - op("N", s1)
    return sz
end

function string_order(psi::MPS, i, j)
    sites = siteinds(psi)
    os = op("Sz",sites[i])
    for k in (i+1):(j-1)
        os *= exp(1im * pi * op("Sz", sites[k]))
    end
    os *= op("Sz", sites[j])
    return inner(psi, apply(os, psi))
end

function string_order_2(psi::MPS, i, j)
    sites = siteinds(psi)
    os = [op("Sz", sites[i])]
    a=ops([("Sz",i) for i in (i+1):(j-1)], sites)
    append!(os, exp.(1im*pi*a))
    append!(os, [op("Sz", sites[j])])
    return inner(psi, apply(os, psi))
end

function structure_factor(k, psi::MPS)
    C = correlation_matrix(psi, "Sz", "Sz")
    ft = 0
    for i in 1:length(psi)
        for j in 1:length(psi)
            ft += C[j, i]*exp(1im*k*(j-i))
        end
    end
    return ft/length(psi)
end

function momentum_distribution(k, psi::MPS)
    C = correlation_matrix(psi, "Adag", "A")
    ft = 0
    for i in 1:length(psi)
        for j in 1:length(psi)
            ft += C[j, i]*exp(1im*k*(j-i))
        end
    end
    return ft/length(psi)
end

function local_correlations(N,sites)
    C2 = OpSum()
    C3 = OpSum()
    for n in 1:N
        C2 += 1, "N", n, "N", n #on-site interaction
        C2 += -1, "N", n 
        C3 += 1, "N", n, "N", n, "N", n
        C3 += -3, "N", n, "N", n
        C3 += 2, "N", n
    end
    return MPO(C2, sites), MPO(C3, sites)
end

function structure_factor(k, C)
    str = 0. + 1. *im
    for i in 20:(length(psi)-20), j in 20:(length(psi)-20)
        str += exp(1im*k*(i-j))*C[i,j]
    end
    return str/length(C[1:end,1])
end

function bow_ord(C)
    ord = 0
    for i in 1:length(C[1:(end-2),1])
        ord += C[i,i+1]+C[i+1,i]+C[i+1,i+2]+C[i+2,i+1]
    end
    return ord/length(C[1:(end-2),1])
end

function bow_ord(psi::MPS)
    bow = OpSum()
    for i in 1:1
        bow += "Adag", i, "A", (i+1) 
        bow += "Adag", (i+1), "A", (i)
    end
    for i in 1:1
        bow += "Adag", (i+1), "A", (i+2) 
        bow += "Adag", (i+2), "A", (i+1)
    end
    bow_mpo = MPO(bow, siteinds(psi))
    return bow_mpo
end

function dim_ord(i,psi::MPS)
    bow = OpSum()
    bow += "Adag", i, "A", (i+1) 
    bow += "Adag", (i+1), "A", (i)
    #bow += "Adag", (i+1), "A", (i+2) 
    #bow += "Adag", (i+2), "A", (i+1)
    bow_mpo = MPO(bow, siteinds(psi))
    return bow_mpo
end


function chir_ord(i,j, GS)
    ki=2im*(op("Adag",siteinds(GS)[i])*op("A",siteinds(GS)[i+1])-op("A",siteinds(GS)[i+1])*op("Adag",siteinds(GS)[i]))
    kj=2im*(op("Adag",siteinds(GS)[j])*op("A",siteinds(GS)[j+1])-op("A",siteinds(GS)[j+1])*op("Adag",siteinds(GS)[j]))
    return ki*kj
end

function chir_ord(i, j, psi)
    corr = OpSum()
    corr += -4, "Adag", i, "A", (i+1), "Adag", j, "A", (j+1)
    corr += +4, "Adag", i, "A", (i+1), "A", j, "Adag", (j+1)
    corr += +4, "A", i, "Adag", (i+1), "Adag", j, "A", (j+1)
    corr += -4, "A", i, "Adag", (i+1), "A", j, "Adag", (j+1)
    corr_op = MPO(corr, siteinds(psi))
    return corr_op
end

function current(i,psi)
    cur = OpSum()
    cur += 2im, "Adag", i, "A", (i+1)
    cur += -2im, "A", (i+1), "Adag", i
    op = MPO(cur, siteinds(psi))
    return op
end


function dmrg_run(N, J1, J2, U; loc_hilb=4, sparse_multithreading = false)
    
    if sparse_multithreading
        ITensors.Strided.set_num_threads(1)
        BLAS.set_num_threads(1)
        ITensors.enable_threaded_blocksparse()
    else
        ITensors.Strided.set_num_threads(16)
        BLAS.set_num_threads(16)
        ITensors.disable_threaded_blocksparse()
    end

    sites = siteinds("Boson", N; dim=loc_hilb, conserve_qns=true)

    #=set up of DMRG schedule=#
    sweeps = Sweeps(18)
    setmaxdim!(sweeps, 200, 200, 200, 200, 300, 300, 300, 400, 400, 500, 600)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 0)
    en_obs = DMRGObserver(energy_tol = 1e-4)

    in_state = [isodd(n) ? "1" : "0" for n in 1:N] #half filling
    psi0 = randomMPS(sites, in_state, linkdims=100)

    #= hamiltonian definition =#
    ampo = OpSum()  
    for b1 in 1:(N-1)
        ampo += -J1*(-1)^(b1-1), "Adag", b1, "A", (b1+1)
        ampo += -J1*(-1)^(b1-1), "Adag", (b1+1), "A", b1
    end
    for b1 in 1:(N-2)
        ampo += -J2*exp(1im*10e-8), "Adag", b1, "A", (b1+2)
        ampo += -J2*exp(-1im*10e-8), "Adag", (b1+2), "A", b1
    end
    for n in 1:N
        ampo += U/2, "N", n, "N", n #on-site interaction
        ampo += -U/2, "N", n 
    end

    H = MPO(ampo, sites)

    if sparse_multithreading
        H = splitblocks(linkinds,H)
    end



    ####################################
    #= DMRG =#
    energy, psi = @time dmrg(H, psi0, sweeps, observer = en_obs, verbose=false);
    ####################################

    #= write mps to disk =#
    #f = h5open("MPS_HDF5/local.h5","cw")
    #write(f,"Nx=($Nx)_Ny=($Ny)",psi)
    #close(f)

    H2 = inner(H, psi, H, psi)
    variance = H2-energy^2
    @show real(variance) #variance to check for convergence

    ITensors.disable_threaded_blocksparse()
    ITensors.Strided.set_num_threads(16)
    BLAS.set_num_threads(16)

    #= observables computation =#
        
    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix
    lutt = zeros(N-11); chir = ones(N-11)
    for j in 2:(N-10-1)
        lutt[j] = real(C[10,10+j])
        #chir[j] = real(inner(psi,apply(chir_ord(5,5+j,psi),psi)))
        chir[j] = real(inner(psi,chir_ord(10,10+j,psi),psi))
    end
    bow = bow_ord(C)

    dens = expect(psi, "N"); #density

    @show sum(dens)

    fig, ax = plt.subplots(2,1, figsize=(8,5))

    C2, C3 = local_correlations(N,sites) #local correlation operators
    exp_C2 = @show real(inner(psi,C2,psi)/sum(dens))
    exp_C3 = @show real(inner(psi,C3,psi)/sum(dens))

    return lutt, bow, chir, psi, C
end

U = 1; loc_hilb = 4; N = 120
let
    for J2 in 0.2:0.1:1
        lutt, bow, chir, psi, C = dmrg_run(N, 1, J2, U, loc_hilb=loc_hilb, sparse_multithreading = false) #worth setting sparse_multithreading to false if Ny <4
        
        open("U($U)_N($N)_dim($loc_hilb).txt", "a") do io
            writedlm(io, [real(J2) real(abs.(lutt[end])) real(bow) real(abs.(chir[end])) real(abs.(lutt[end-1]))])
        end

        f = h5open("MPS_HDF5/QFM.h5","cw")
        write(f,"U($U)_N($N)_dim($loc_hilb)_J2($J2)",psi)
        close(f)
    end
end

let
    f = h5open("data2.h5","r")
    psi = read(f,keys(f)[end-1],MPS)

    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix
    lutt = real(C[20,N-20])
    chir = real(inner(psi,chir_ord(20,N-20,psi),psi))
    bow = @show inner(psi,bow_ord(psi),psi)/(length(psi)-40)

    open("U($U)_N($N)_dim($loc_hilb).txt", "a") do io
        writedlm(io, [real(J2) real(abs.(lutt)) real(bow) real(abs.(chir))])
    end

    f = h5open("MPS_HDF5/QFM_dim4_U$U.h5","cw")
    write(f,"U($U)_N($N)_dim($loc_hilb)_J2($J2)",psi)
    close(f)

    plt.plot(dim)
    close(f)
end

U = 3; loc_hilb = 4; N = 140; J2=0.4
let
        matr = readdlm("data_1D/U(6)_N(140)_dim(4)_V4.txt", '\t')
        println(matr)
        js = matr[1:end,1]

        fig, ax = plt.subplots(3,1, figsize=(6,6), tight_layout = true)

        #ax[1].set_title("N=$N, U=$U, dim=$loc_hilb")
        ax[1].plot(js, matr[1:end,2], "--o", color = "tab:blue")
        ax[1].set_ylabel(L"$g^1(|i-j|)$"); #ax[1].grid(true)

        ax[2].plot(js, matr[1:end,3]*N/(N-40), "--s", color="tab:orange")
        ax[2].set_ylabel(L"$\Delta B$"); #ax[2].grid(true)

        ax[3].plot(js, matr[1:end,4], "--^", color="tab:green")
        ax[3].set_ylabel(L"$ \kappa ^2(|i-j|)$")
        ax[3].set_xlabel("U/J₁"); #ax[3].grid(true)

    plt.savefig("plots/hardcore_phasediag.pdf")
end

#plot phase diagram
matr = readdlm("data_1D/phase_diag.txt", '\t')
println(matr)
fig, ax = plt.subplots(1,1, figsize=(4,4), tight_layout = true)
#ax.set_title("struc order, N=$N, U=$U, dim=$loc_hilb, J2=$J2")
#ax.plot(range(0,2pi,length=300)/pi,abs.(matr[:,1]), "-o", color="b")
x1 = [0.9,matr[:,1]...,6.1]; y1 =[0.3,matr[:,2]...,0.34]; y2 = [0.3,matr[:,3]...,0.45]
ax.plot(matr[:,1],matr[:,2], "-o", color="tab:blue")
ax.plot(matr[:,1],matr[:,3], "-o", color="tab:blue")
ax.fill_between(x1, y1, y2, color="C0", alpha=0.3)
plt.text(3, 0.31, "SF", fontsize=20)
plt.text(4.5, 0.39, "BOW", fontsize=20)
plt.text(1.5, 0.42, "CSF", fontsize=20)
ax.fill_between(x1, 0.28 .*fill(1,length(x1)), y1, color="C1", alpha=0.3)
ax.fill_between(x1, y2, 0.46 .*fill(1,length(x1)), color="C2", alpha=0.3)
ax.set_xlim(0.9,6.1)
ax.set_ylim(0.29,0.46)
ax.set_ylabel("J₂/J₁"); #ax.grid(true)
ax.set_xlabel("U/J₁");
plt.savefig("plots/phase_diag.pdf")
#ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g π"))
#ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(base=0.5))
#plt.savefig("plots/phase_diag.pdf")



#observable_computation(GS, 16, 2, μ = 0.0, t = 1.0, U = 4.0, α = 1//6, ν = 1/2)

# fanno misure molto precise sulla densità, bisogna fare correlatore n_i n_j, 
# fissando i=20 e j=n-20, questo deve decadere come power law nei superfluidi e 
# vedere che succede nella bow, ma la trasformata di fourier (structure factor)
# dovrebbe dare un picco a k finito

#fissi j2/j1 a 0.4 e salgo U/J1
#calcolare lo string order parameter (quello even del paper di sergi)
#nei sistemi peierls instability può essere che ci siano difetti 

#DOMANDE
# somma operatori sx(1)*id+sx(2)*id=sx(1)*sx(2)
# a cosa corrisponde intuitivamente "minimizzare l'hopping"?

