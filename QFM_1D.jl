#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
tkz = pyimport("tikzplotlib")
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5
using LaTeXStrings, DelimitedFiles
using FFTW

function ITensors.op(::OpName"Sz", ::SiteType"Boson", s1::Index) #Sz operator equivalent for hardcore bosons
    sz = -1/2*op("Id", s1) + op("N", s1)
    return sz
end

function ITensors.op(::OpName"SzSz1", ::SiteType"Boson", s1::Index, s2::Index) #Sz operator equivalent for hardcore bosons
    szsz1 = op("Sz", s1)*op("Id", s2) + op("Id", s1)*op("Sz", s2)
    return szsz1
end

function string_order_2(psi::MPS, i, j)
    sites = siteinds(psi)
    os = [op("Sz", sites[i])]
    a=ops([("Sz",k) for k in (i+1):1:(j-1)], sites)
    append!(os, exp.(1im*pi*a))
    append!(os, [op("Sz", sites[j])])
    return inner(psi, apply(os, psi))
end

function structure_factor(k, psi::MPS) #fourier transform of density-density (captures )
    C = correlation_matrix(psi, "Sz", "Sz")
    ft = 0
    for i in 1:length(psi)
        for j in 1:length(psi)
            ft += C[j, i]*exp(1im*k*(j-i))
        end
    end
    return ft/length(psi)
end

function local_correlations(N,sites) #same site two and three particles correlator MPO
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

function bow_ord(C) #BOW order parameter form correlation function
    ord = 0
    for i in 1:length(C[1:(end-2),1])
        ord += C[i,i+1]+C[i+1,i]+C[i+1,i+2]+C[i+2,i+1]
    end
    return ord/length(C[1:(end-2),1])
end

function bow_ord(psi::MPS) #BOW order parameter from MPS
    bow = OpSum()
    N = length(psi)
    for i in 20:(N-20)
        bow += "Adag", i, "A", (i+1) 
        bow += "Adag", (i+1), "A", (i)
    end
    for i in 20:(N-20)
        bow += "Adag", (i+1), "A", (i+2) 
        bow += "Adag", (i+2), "A", (i+1)
    end
    bow_mpo = MPO(bow, siteinds(psi))
    return bow_mpo
end

function charge_order(psi)
    C = correlation_matrix(psi, "Sz", "Sz")
    Cd = correlation_matrix(psi, "Adag", "A")
    ch_ord = zeros(ComplexF64,length(psi)-100)
    g1 = zeros(ComplexF64,length(psi)-100)
    for i in 1:(length(psi)-100)
        ch_ord[i] = C[50,50+i]
    end
    return diag(Cd), ch_ord
end

function dim_ord(i,psi::MPS) #dimerization order parameter
    bow = OpSum()
    bow += "Adag", i, "A", (i+1) 
    bow += "Adag", (i+1), "A", (i)
    #bow += "Adag", (i+1), "A", (i+2) 
    #bow += "Adag", (i+2), "A", (i+1)
    bow_mpo = MPO(bow, siteinds(psi))
    return bow_mpo
end

function chir_ord(i,j, GS) #CSF order parameter from ground state
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

function current(i,psi) #current operator
    cur = OpSum()
    cur += 2im, "Adag", i, "A", (i+1)
    cur += -2im, "A", (i+1), "Adag", i
    op = MPO(cur, siteinds(psi))
    return op
end

function momentum_distribution(k, C) #fourier transfor of hopping
    ft = 0.0 + 0im
    for i in 50:1:(length(C[:,1])-50)
        for j in 50:1:(length(C[:,1])-50)
            #ind = ind - (length(C[:,1])-100)/2 -1
            #println(ind)
            ind = j-i
            #sin(pi*(ind)/(length(C[:,1])-101))^2*
            ft += (abs(real(C[i, j])))*exp(1im*k*(ind))
        end
    end
    return ft/(length(C[:,1])-100)
end
###########################################3

function dmrg_run(N, J1, J2, U, V; loc_hilb=4, sparse_multithreading = false, psi0 = nothing, max_linkdim = 300)
    
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
    sweeps = Sweeps(20)
    setmaxdim!(sweeps, 100, 100, 200, 200, 200, 200, max_linkdim)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 0)
    en_obs = DMRGObserver(energy_tol = 1e-6)

    in_state = [isodd(n) ? "1" : "0" for n in 1:N] #half filling
    if psi0 === nothing
        psi0 = randomMPS(sites, in_state, linkdims=100)
    else 
        sites = siteinds(psi0)
    end

    #= hamiltonian definition =#
    ampo = OpSum()  
    for b1 in 1:(N-1)
        ampo += -J1*(-1)^(b1-1), "Adag", b1, "A", (b1+1)
        ampo += -J1*(-1)^(b1-1), "Adag", (b1+1), "A", b1
        ampo += V, "N", b1, "N", (b1+1) #dens dens interaction
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
    
    return psi
end

function main_dmrg(J2,U,V,N,loc_hilb; reupload = false)
    if reupload;
        f = h5open("MPS.h5","r")
        psi0 = read(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V",MPS)
        close(f)
    else
        psi0 = nothing
    end

    println("DMRG run: #threads=$(Threads.nthreads()), U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)")

    psi = dmrg_run(N, 1, J2, U, V, loc_hilb = loc_hilb, sparse_multithreading = true, psi0 = psi0)

    h5open("MPS.h5","cw") do f
        if haskey(f, "psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V")
            delete_object(f, "psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V")
        end
        write(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V",psi)
    end
end

function main_obs(J2,U,V,N,loc_hilb) #computation of observables
    f = h5open("MPS.h5", "r")
    psi = read(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V", MPS)
    close(f)   
    
    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix
    #CSF = zeros(ComplexF64, length(psi)-100) 
    #for i in 1:(length(psi)- 100)
    #    CSF[i] = inner(psi, chir_ord(50,49+i,psi),psi)
    #end

    #lutt = sum(abs.(C[50,(end-60):(end-50)]))/10 #luttinger liquid parameter
    #chir = real(inner(psi, chir_ord(50,N-50,psi),psi)) #chiral order parameter
    bow = @show inner(psi,bow_ord(psi),psi)/(length(psi)-100)
    #lutt=0; bow=0
    
    dim = zeros(length(psi))
    for i in 1:length(psi)
        dim[i] = (1)^(i-1)*real(C[i,i])
    end
    dim_par = sum(dim)/length(psi)
    plt.plot(dim)
    println(dim)
    
    open("data_1D/cdw_bow_U($U)_N($N)_dim($loc_hilb)_V.txt", "a") do io
        writedlm(io, [real(V) real(J2) real(dim_par) real(bow)])
    end

    #=
    h5open("corr_QFM.h5","cw") do f
        if haskey(f, "CSF_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
            delete_object(f, "CSF_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
            delete_object(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
        end
        write(f,"CSF_U($U)_N($N)_J2($J2)_dim($loc_hilb)",CSF)
        write(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)",C)
    end
    =#
    
end

function main_processing(U,N,loc_hilb) #plots
    matr = readdlm("data_1D/cdw_bow_U($U)_N(250)_dim($loc_hilb)_V.txt", '\t')
    println(matr)
    
    js = matr[1:9,1]

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 100)
    
    #ax[1].set_title("N=$N, U=$U, dim=$loc_hilb")
    ax.plot(js, matr[1:9,3], "--o", color = "tab:blue", label = "CDW")
    par1 = ax.twinx()
    par1.plot(js, abs.(matr[1:9,4])*(N)/(N-40), "--s", color = "tab:orange", label = "BOW")
    ax.set_xlabel("V/J1"); ax.set_ylabel("CDW")
    par1.set_ylabel("BOW")
    plt.legend(title = "J2=0.4")
    
    #=
    matr = readdlm("data_1D/cdw_bow_U($U)_N(251)_dim($loc_hilb)_V.txt", '\t')
    println(matr)
    js = matr[8:14,1]

    ax[1].plot(js, matr[8:14,3], "--o", color = "tab:blue", label = "CDW")
    par1 = ax[1].twinx()
    par1.plot(js, abs.(matr[8:14,4])*(N)/(N-40), "--s", color = "tab:orange", label = "BOW")
    ax[1].set_xlabel("V/J1"); ax[1].set_ylabel("CDW")
    par1.set_ylabel("BOW")
    plt.legend(title = "J2=0.4")
    =#
    #ax[1].plot(js, matr[11:15,3], "--^", color = "tab:red", label = "0.4")
    #ax[1].set_ylabel(L"$g^1(|i-j|)$"); #ax[1].grid(true)

    #js = matr[15:21,2]

    #=
    ax[2].plot(js, matr[15:21,3] , "--o", color = "tab:blue", label = "CDW")
    par1 = ax[2].twinx()
    par1.plot(js, abs.(matr[15:21,4])*(N)/(N-40), "--s", color = "tab:orange", label = "BOW")
    ax[2].set_xlabel("J2/J1"); ax[2].set_ylabel("CDW")
    par1.set_ylabel("BOW")
    plt.legend(title = "V=1.5")
    #ax[2].plot(js, matr[11:15,4]*N/(N-40), "--^", color = "tab:red", label = "0.4")
    #ax[1].set_ylabel(L"$\Delta B$"); #ax[2].grid(true)
    =#

    #matr = readdlm("data_1D/U($U)_N($N)_dim($loc_hilb)_V.txt", '\t')
    #ax[3].plot(js, matr[1:5,3], "--o", color = "tab:blue", label = "0.2")
    #ax[3].plot(js, matr[6:10,3], "--s", color = "tab:orange", label = "0.3")
    #ax[3].plot(js, matr[11:15,3], "--^", color = "tab:red", label = "0.4")

    #ax[3].plot(js, matr[1:end,4], "--^", color="tab:green")
    #ax[3].set_ylabel(L"$ \langle S_z S_z \rangle$")
    #ax[3].set_xlabel("V/J₁"); #ax[3].grid(true)
    #plt.legend(title = "J₂/J₁")
    #tkz.save("plots/U6_phasediag.tex")
    plt.savefig("plots/cdw_bow_U($U)_N(250)_dim($loc_hilb)_V.pdf")
end

#############################################

function plot_mom_dist(J2,U,N,loc_hilb, m, bz)
    f = h5open("MPS.h5", "r")
    psi = read(f,"psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)", MPS)
    close(f)   
    
    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix

    mom_d = zeros(ComplexF64, m)
    for (i,k) in enumerate(-(bz*pi):((2*bz*pi)/m):(bz*pi))
        mom_d[i] = momentum_distribution(k,C)
    end

    h5open("corr_QFM.h5","cw") do f
        if haskey(f, "mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
            delete_object(f, "mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
            delete_object(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
        end
        write(f,"mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)",mom_d)
        write(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)",C)
    end
    
end

function plot_mom_dist_2(J2,U,N,loc_hilb, bz)
    f = h5open("corr_QFM.h5", "r")
    mom_d = read(f,"mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
    C = read(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
    close(f)   

    m =length(mom_d)
    mom_d = zeros(ComplexF64, m)
    for (i,k) in enumerate(-(bz*pi):((2*bz*pi)/m):(bz*pi))
        mom_d[i] = momentum_distribution(k,abs.(real(C)))
        if real(mom_d[i]) > 10
            println(k)
        end
    end
    @show(C)

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 150)
    ax.plot(-(bz*pi):((2*bz*pi)/m):(bz*pi), real(mom_d), "--o", color = "tab:blue")
    ax.set_title("mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)"); #ax[1].grid(true)
    ax.set_ylabel(L"Momentum distribution"); #ax[1].grid(true)
    #ax.set_xscale()
    ax.set_xlabel("k"); #ax[3].grid(true)
    plt.savefig("mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)_$bz.pdf")

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 150)
    #ax.plot(real(C[50,50:(end-50)])) #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax.plot(imag(C[50,50:(end-50)])) #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    ax.plot(angle.(C[50,50:(end-50)])) #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax.plot((C[50,50:(end-50)] .+ C[50:(end-50), 50]) ./2 ) #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    asc = collect(0:0.01:150)
    #plt.plot(asc,0.1 .* cos.((-0.2827433388230811) .* asc) .+0.1 .* cos.((pi-pi) .* asc), label = "no-pi")    
    ax.set_title("corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)"); #ax[1].grid(true)
    ax.set_ylabel(L"Correlation function"); #ax[1].grid(true)
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_xlabel("x"); #ax[3].grid(true)
    plt.savefig("corr_U($U)_N($N)_J2($J2)_dim($loc_hilb).pdf")
end

function plot_mom_dist_3(J2,U,N,loc_hilb, bz)
    #fig1, ax1 = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 150)

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 200)
    f = h5open("corr_QFM.h5", "r")
    mom_d1 = read(f,"mom_U($U)_N($N)_J2(0.1)_dim($loc_hilb)")
    C1 = read(f,"corr_U($U)_N($N)_J2(0.1)_dim($loc_hilb)")
    mom_d2 = read(f,"mom_U($U)_N($N)_J2(0.4)_dim($loc_hilb)")
    C2 = read(f,"corr_U($U)_N($N)_J2(0.4)_dim($loc_hilb)")
    mom_d3 = read(f,"mom_U($U)_N($N)_J2(0.8)_dim($loc_hilb)")
    C3 = read(f,"corr_U($U)_N($N)_J2(0.8)_dim($loc_hilb)")
    close(f)   

    m =length(mom_d1)

    fig, ax = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 250)
    ax.plot((-(bz*pi):((2*bz*pi)/m):(bz*pi) )./pi, real(mom_d1)./150, "-.", markersize = 3., color = "tab:blue", label = "0.1")
    ax.plot((-(bz*pi):((2*bz*pi)/m):(bz*pi) )./pi, real(mom_d2)./150, "--", markersize = 3., color = "tab:orange", label = "0.4")
    ax.plot((-(bz*pi):((2*bz*pi)/m):(bz*pi) )./pi, real(mom_d3)./150, "-", markersize = 3., color = "tab:red", label = "0.8")
    #ax.set_title("mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)"); #ax[1].grid(true)
    plt.legend(title = L"J_2/J_1")
    ax.set_ylabel("Momentum distribution"); #ax[1].grid(true)
    #ax.set_xscale()
    ax.set_xlabel("k"); #ax[3].grid(true)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g π"))
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(base=0.5))
    tkz.save("plots/comp_mom_U($U)_N($N)_J2($J2)_dim($loc_hilb).tex")
    plt.savefig("plots/comp_mom_U($U)_N($N)_J2($J2)_dim($loc_hilb).pdf")

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    ax1.plot(abs.(C1[50,50:(end-50)]), label = "SF", color = "tab:blue") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    ax1.plot(abs.(C2[50,50:(end-50)]), label = "BOW", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    ax1.plot(abs.(C3[50,50:(end-50)]), label = "CSF", color = "tab:green") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    plt.legend()
    #ax.plot((C[50,50:(end-50)] .+ C[50:(end-50), 50]) ./2 ) #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax.set_title("corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)"); #ax[1].grid(true)
    ax1.set_ylabel("Correlation function"); #ax[1].grid(true)
    ax1.set_ylim(1e-7,1e-0)
    ax1.set_xlim(0,100)
    #ax.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    #plt.savefig("comp_C_U($U)_N($N)_J2($J2)_dim($loc_hilb).pdf")
    #plt.show()
end

function plot_mom_dist_4(J2,U,N,loc_hilb, bz)
    #fig1, ax1 = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 150)

    f = h5open("corr_QFM.h5", "r")
    mom_d1 = read(f,"mom_U($U)_N($N)_J2(0.1)_dim($loc_hilb)")
    C1 = read(f,"corr_U($U)_N($N)_J2(0.1)_dim($loc_hilb)")
    mom_d2 = read(f,"mom_U($U)_N($N)_J2(0.5)_dim($loc_hilb)")
    C2 = read(f,"corr_U($U)_N($N)_J2(0.5)_dim($loc_hilb)")
    close(f)   

    m =length(mom_d1)

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    ax1.plot(real.(C1[50,50:(end-50)]), label = "Real part", color = "tab:blue") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    ax1.plot(abs.(C1[50,50:(end-50)]), label = "Abs. value", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    plt.legend()
    ax1.set_ylabel("Correlation function"); #ax[1].grid(true)
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    ax1.set_title("SF")
    ax1.set_xlim(-1,100)
    #plt.savefig("comp_Cabs_U($U)_N($N)_J2(0.1)_dim($loc_hilb).pdf")
    println(C2[50,50:(end-50)])

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    ax1.plot(abs.(C2[50,50:(end-50)]), label = "Real part", color = "tab:green") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    ax1.plot(imag.(C2[50,50:(end-50)]), label = "Imag. part", color = "tab:orange") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    ax1.plot(real.(C2[50,50:(end-50)]), label = "Abs. value", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    plt.legend()
    ax1.set_title("CSF")
    ax1.set_ylabel("Correlation function"); #ax[1].grid(true)
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    ax1.set_xlim(-1,100)
    #plt.savefig("comp_temp_U($U)_N($N)_J2(0.5)_dim($loc_hilb).pdf")
end

function plot_obs_U1()
    U = 0.25; loc_hilb = 4; N = 250
    f = h5open("corr_QFM.h5", "r")
    CSF1 = read(f,"CSF_U($U)_N($N)_J2(0.24)_dim($loc_hilb)")
    C1 = read(f,"corr_U($U)_N($N)_J2(0.24)_dim($loc_hilb)")
    CSF2 = read(f,"CSF_U($U)_N($N)_J2(0.26)_dim($loc_hilb)")
    C2 = read(f,"corr_U($U)_N($N)_J2(0.26)_dim($loc_hilb)")
    CSF3 = read(f,"CSF_U($U)_N($N)_J2(0.28)_dim($loc_hilb)")
    C3 = read(f,"corr_U($U)_N($N)_J2(0.28)_dim($loc_hilb)")
    CSF4 = read(f,"CSF_U($U)_N($N)_J2(0.3)_dim($loc_hilb)")
    C4 = read(f,"corr_U($U)_N($N)_J2(0.3)_dim($loc_hilb)")
    CSF5 = read(f,"CSF_U($U)_N($N)_J2(0.34)_dim($loc_hilb)")
    C5 = read(f,"corr_U($U)_N($N)_J2(0.34)_dim($loc_hilb)")
    CSF6 = read(f,"CSF_U($U)_N($N)_J2(0.36)_dim($loc_hilb)")
    C6 = read(f,"corr_U($U)_N($N)_J2(0.36)_dim($loc_hilb)")
    close(f)   
    println(length(C1[50,50:(end-50)]))

    fig, ax = plt.subplots(1,2, figsize=(4,4), tight_layout = true, dpi = 250)
    ax[1].plot(0:150, abs.(C1[50,50:(end-50)]), "--", color = "tab:blue", label = "0.24")
    ax[1].plot(0:150, abs.(C2[50,50:(end-50)]), ".", color = "tab:green", label = "0.26")
    ax[1].plot(0:150, abs.(C3[50,50:(end-50)]), "-.", color = "tab:olive", label = "0.28")
    ax[1].plot(0:150, abs.(C4[50,50:(end-50)]), "-", color = "tab:cyan", label = "0.32")
    #ax[1].plot(0:150, abs.(C5[50,50:(end-50)]), "-", color = "tab:red", label = "0.34")
    #ax.plot(0:150, abs.(C6[50,50:(end-50)]), "-", color = "tab:purple", label = "0.36")
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    ax[1].set_ylabel(L"g^1(|i-j|)")
    ax[1].set_xlabel(L"|i-j|"); #ax[3].grid(true)

    ax[1].legend(title=L"J_2/J_1")

    #tkz.save("prova.tex")
    #plt.savefig("C_decay.pdf")

    #fig, ax = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 250)
    ax[2].plot(0:149, abs.(CSF1), "--", color = "tab:blue", label = "0.24")
    ax[2].plot(0:149, abs.(CSF2), ".", color = "tab:green", label = "0.26")
    ax[2].plot(0:149, abs.(CSF3), "-.", color = "tab:olive", label = "0.28")
    ax[2].plot(0:149, abs.(CSF4), "-", color = "tab:cyan", label = "0.32")
    #ax[2].plot(0:149, abs.(CSF5), "-", color = "tab:red", label = "0.34")
    #ax[2].plot(0:149, abs.(CSF6), "-", color = "tab:purple", label = "0.36")
    ax[2].set_ylabel(L"\kappa ^2(|i-j|)")
    #ax[2].set_xscale("log")
    #ax[2].set_yscale("log")
    ax[2].set_xlabel(L"|i-j|"); #ax[3].grid(true)
    ax[2].legend(title=L"J_2/J_1")
    
    tkz.save("decay.tex")
    #plt.savefig("CSF_decay.pdf")

    #ax.set_xscale()
    ax[2].set_xlabel(L"|i-j|"); #ax[3].grid(true)

    #plt.savefig("comp_mom_U($U)_N($N)_J2($J2)_dim($loc_hilb).pdf")
end

function plot_string(J2,U,V,N,loc_hilb)
    f = h5open("MPS.h5", "r")
    psi1 = read(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V", MPS)
    close(f)

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    
    str_ord_odd = zeros(length(psi1)-200)
    str_ord_even = zeros(length(psi1)-200)
    for i in 1:length(str_ord_even)
        j = 50+2*(i-1)+1
        println(j)
        str_ord_odd[i] = abs.(string_order_2(psi1, 50, j))
        #str_ord_even[i] = abs.(string_order_2(psi1, 50, j-1))
    end
    println(str_ord_odd)
    ax1.plot(1:2:2*length(str_ord_odd),str_ord_odd, "-.", label = "$V", color = "tab:blue") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    

    open("data_1D/string_even_U($U)_N($N)_dim($loc_hilb)_V.txt", "a") do io
        writedlm(io, [real(V) real(J2) str_ord_odd[end]])
    end
    #=
    str_ord_odd = zeros(length(psi1)-200)
    str_ord_even = zeros(length(psi1)-200)
    for i in 1:length(str_ord_even)
        j = 50+2*(i-1)
        println(j)
        str_ord_odd[i] = real.(string_order_2(psi2, 50, j))
        #str_ord_even[i] = abs.(string_order_2(psi1, 50, j-1))
    end
    println(str_ord_even)
    ax1.plot(1:2:2*length(str_ord_odd), str_ord_odd, "--", label = "0.4", color = "tab:orange") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    
    str_ord_odd = zeros(length(psi1)-200)
    str_ord_even = zeros(length(psi1)-200)
    for i in 1:length(str_ord_even)
        j = 50+2*(i-1)
        println(j)
        str_ord_odd[i] = real.(string_order_2(psi3, 50, j))
        #str_ord_even[i] = abs.(string_order_2(psi1, 50, j-1))
    end
    println(str_ord_even)
    ax1.plot(1:2:2*length(str_ord_odd),str_ord_odd, "-", label = "0.6", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    
    plt.legend(title = L"J_2/J_1")
    #plt.text(10, 0.03, "SF", fontsize=15)
    #plt.text(45, 0.16, "BOW", fontsize=15)
    #plt.text(40, 0.015, "CSF", fontsize=15)
    =#
    plt.title("J2 = $J2, U = $U, N = $N, dim = $loc_hilb")
    plt.legend(title = L"J_2/J_1")
    ax1.set_ylabel("String order"); #ax[1].grid(true)
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    plt.savefig("plots/string_even_U($U)_J2($J2)_N($N)_dim($loc_hilb)_V.pdf")
    #ax1.set_xlim(-1,100)
    #tkz.save("string_U($U)_N($N)_J2(0.1)_dim($loc_hilb).tex")
    
end

function plot_string_2(J2,U,V,N,loc_hilb)
    matr = readdlm("data_1D/string_even_U($U)_N($N)_dim($loc_hilb)_V.txt", '\t')
    println(matr)
    js = matr[1:5,1]

    fig, ax = plt.subplots(3,1, figsize=(6,6), tight_layout = true, dpi = 100)

    #ax[1].set_title("N=$N, U=$U, dim=$loc_hilb")
    ax[1].plot(js, matr[1:5,3], "--o", color = "tab:blue", label = "0.2")
    ax[1].plot(js, matr[6:10,3], "--s", color = "tab:orange", label = "0.3")
    ax[1].plot(js, matr[11:15,3], "--^", color = "tab:red", label = "0.4")
    ax[1].set_ylabel(L"$Even String$"); #ax[1].grid(true)

    matr = readdlm("data_1D/string_odd_U($U)_N($N)_dim($loc_hilb)_V.txt", '\t')

    ax[2].plot(js, matr[1:5,3]*N/(N-40) , "--o", color = "tab:blue", label = "0.2")
    ax[2].plot(js, matr[6:10,3]*N/(N-40), "--s", color = "tab:orange", label = "0.3")
    ax[2].plot(js, matr[11:15,3]*N/(N-40), "--^", color = "tab:red", label = "0.4")
    ax[2].set_ylabel(L"$Odd string$"); #ax[2].grid(true)

    matr = readdlm("data_1D/string_sz1sz2_U($U)_N($N)_dim($loc_hilb)_V_1.txt", '\t')
    
    ax[3].plot(js, matr[1:5,3], "--o", color = "tab:blue", label = "0.2")
    ax[3].plot(js, matr[6:10,3], "--s", color = "tab:orange", label = "0.3")
    ax[3].plot(js, matr[11:15,3], "--^", color = "tab:red", label = "0.4")

    #ax[3].plot(js, matr[1:end,4], "--^", color="tab:green")
    ax[3].set_ylabel(L"$VBS string$")
    ax[3].set_xlabel("V/J₁"); #ax[3].grid(true)
    plt.legend(title = "J₂/J₁")
    #tkz.save("plots/U6_phasediag.tex")
    plt.savefig("plots/string_V.pdf")
    
end

function plot_V(J2,U,V,N,loc_hilb)
    f = h5open("MPS.h5", "r")
    psi = read(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V", MPS)
    #psi2 = read(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb)_V", MPS)
    close(f)
    
    C1, dw = charge_order(psi)
    #C2, dw2 = charge_order(psi2)
    #C3, dw3 = charge_order(psi3)

    open("data_1D/U($U)_N($N)_dim($loc_hilb)_V.txt", "a") do io
        writedlm(io, [real(V) real(J2) abs(dw[end])])
    end

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    ax1.plot(abs.(C1), label = "$J2",  ".", color = "tab:green") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax1.plot(abs.(C2), label = "0.4",  ".", color = "tab:blue") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax1.plot(abs.(C3), label = "0.6",  ".", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    plt.legend()
    ax1.set_title("Density (V=$V)")
    ax1.set_ylabel("density"); #ax[1].grid(true)
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    plt.savefig("dens_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb).pdf")

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    ax1.plot(abs.(dw), label = "$J2", color = "tab:green") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax1.plot(abs.(dw2), label = "0.4", color = "tab:blue") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax1.plot(abs.(dw3), label = "0.6", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    plt.legend()
    ax1.set_title("Correlator (V=$V)")
    ax1.set_ylabel("density-density correlation"); #ax[1].grid(true)
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    ax1.set_xlim(-1,100)
    plt.savefig("corr_dens_U($U)_V($V)_N($N)_J2($J2)_dim($loc_hilb).pdf")
end

for J2 in [0.4]
    for V in [1.4,1.6,1.8,2.0,2.2,2.4]
    U = 6; loc_hilb = 4; N = 201; reupload = true; #J2 = 0.1
    main_dmrg(J2,U, V,N,loc_hilb, reupload = reupload)
    #main_obs(J2,U,V,N,loc_hilb)
    #main_processing(U,N, loc_hilb)
    #plot_string(J2,U,V,N,loc_hilb)
    #plot_mom_dist_3(J2,U,N,loc_hilb, 1.)
    #plot_V(J2,U,V,N,loc_hilb)
    end
end

#=
U = 1; loc_hilb = 2; N = 250; reupload = true;
J2 = 1.4
f = h5open("corr_QFM.h5", "r")
C = read(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
close(f)
println(sum(abs.(C[50,(end-60):(end-50)]))/10)
=#

#=
plot_mom_dist(J2,U,N,loc_hilb, 200, 1.)
plot_mom_dist_2(J2,U,N,loc_hilb, 1.)
=#

#=
#plot phase diagram
matr = readdlm("data_1D/phase_diag.txt", '\t')
println(matr)
fig, ax = plt.subplots(1,1, figsize=(3.5,3.1), tight_layout = true, dpi = 200)
#ax.set_title("struc order, N=$N, U=$U, dim=$loc_hilb, J2=$J2")
#ax.plot(range(0,2pi,length=300)/pi,abs.(matr[:,1]), "-o", color="b")
x1 = [0.0,matr[:,1]...,6.1]; y1 =[0.25,matr[:,2]...,0.34]; y2 = [0.25,matr[:,3]...,0.45]
ax.plot(matr[:,1],matr[:,2], "o", color="black", markersize = 3.5)
ax.plot(matr[:,1],matr[:,3], "o", color="black", markersize = 3.5)
ax.plot(0,0.25, "o", color = "red", markersize = 3.5)
ax.fill_between(x1, y1, y2, color="C0", alpha=0.3)
plt.text(3, 0.2, "SF", fontsize=15)
plt.text(4.5, 0.38, "BOW", fontsize=15)
plt.text(1.5, 0.5, "CSF", fontsize=15)
ax.fill_between(x1, 0.1 .*fill(1,length(x1)), y1, color="C1", alpha=0.3)
ax.fill_between(x1, y2, 0.8 .*fill(1,length(x1)), color="C2", alpha=0.3)
ax.set_xlim(0.0,6.1)
ax.set_ylim(0.1,0.6)
ax.set_ylabel("J₂/J₁"); #ax.grid(true)
ax.set_xlabel("U/J₁");
tkz.save("plots/phase_diag.tex")
plt.savefig("plots/phase_diag.pdf")
=#

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

#=
function FT(k, C) #fourier transfor of hopping
    ft = 0.0 + 0im
    for i in 1:5:length(C)
        ft += C[i]*exp(1im*k*(i))
    end
    return ft
end

x = collect(1:1:100)
C= 1 ./(x .+0.000001) .* cos.(x)
plt.plot(C)
mom_d = zeros(ComplexF64, 100)
for (i,k) in enumerate(-pi:(2pi/100):pi)
    mom_d[i] = FT(k,C)
end
plt.plot(-pi:(2pi/100):pi,mom_d)
=#




