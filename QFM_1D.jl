#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, ITensors, ITensors.HDF5
using LaTeXStrings, DelimitedFiles
using FFTW

function ITensors.op(::OpName"Sz", ::SiteType"Boson", s1::Index) #Sz operator equivalent for hardcore bosons
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


function dmrg_run(N, J1, J2, U; loc_hilb=4, sparse_multithreading = false, psi0 = nothing, max_linkdim = 600)
    
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
    sweeps = Sweeps(30)
    setmaxdim!(sweeps, 300, 300, 300, 400, 400, 500, max_linkdim)
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

function main_dmrg(J2,U,N,loc_hilb; reupload = false)
    if reupload;
        f = h5open("MPS.h5","r")
        psi0 = read(f,"psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)",MPS)
        close(f)
    else
        psi0 = nothing
    end

    println("DMRG run: #threads=$(Threads.nthreads()), U($U)_N($N)_J2($J2)_dim($loc_hilb)")

    psi = dmrg_run(N, 1, J2, U, loc_hilb = loc_hilb, sparse_multithreading = true, psi0 = psi0)

    h5open("MPS.h5","cw") do f
        if haskey(f, "psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
            delete_object(f, "psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
        end
        write(f,"psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)",psi)
    end
end

function main_obs(J2,U,N,loc_hilb) #computation of observables
    f = h5open("MPS.h5", "r")
    psi = read(f,"psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)", MPS)
    close(f)   
    
    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix
    lutt = real(C[50,N-50]) #luttinger liquid parameter
    chir = real(inner(psi, chir_ord(50,N-50,psi),psi)) #chiral order parameter
    bow = @show inner(psi,bow_ord(psi),psi)/(length(psi)-100)

    open("data_1D/U($U)_N($N)_dim($loc_hilb).txt", "a") do io
        writedlm(io, [real(J2) real(abs.(lutt)) real(bow) real(abs.(chir))])
    end
end

#=
function main_processing(U,N,loc_hilb) #plots
    matr = readdlm("data_1D/U($U)_N($N)_dim($loc_hilb).txt", '\t')
    println(matr)
    js = matr[1:end,1]

    fig, ax = plt.subplots(3,1, figsize=(6,6), tight_layout = true, dpi = 100)

    #ax[1].set_title("N=$N, U=$U, dim=$loc_hilb")
    ax[1].plot(js, matr[1:end,2], "--o", color = "tab:blue")
    ax[1].set_ylabel(L"$g^1(|i-j|)$"); #ax[1].grid(true)

    ax[2].plot(js, matr[1:end,3]*N/(N-40), "--s", color="tab:orange")
    ax[2].set_ylabel(L"$\Delta B$"); #ax[2].grid(true)

    ax[3].plot(js, matr[1:end,4], "--^", color="tab:green")
    ax[3].set_ylabel(L"$ \kappa ^2(|i-j|)$")
    ax[3].set_xlabel("J₂/J₁"); #ax[3].grid(true)

    plt.savefig("plots/hardcore_phasediag.pdf")


end
=#


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


function momentum_distribution(k, C) #fourier transfor of hopping
    ft = 0.0 + 0im
    for i in 50:1:(length(C[:,1])-50)
        for j in 50:1:(length(C[:,1])-50)
            #ind = ind - (length(C[:,1])-100)/2 -1
            #println(ind)
            ind = j-i
            #sin(pi*(ind)/(length(C[:,1])-101))^2*
            ft += (real.(C[i, j]))*exp(1im*k*(ind))
        end
    end
    return ft/(length(C[:,1])-100)
end

function plot_mom_dist_2(J2,U,N,loc_hilb, bz)
    f = h5open("corr_QFM.h5", "r")
    mom_d = read(f,"mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
    C = read(f,"corr_U($U)_N($N)_J2($J2)_dim($loc_hilb)")
    close(f)   

    m =length(mom_d)
    mom_d = zeros(ComplexF64, m)
    for (i,k) in enumerate(-(bz*pi):((2*bz*pi)/m):(bz*pi))
        mom_d[i] = momentum_distribution(k,C)
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
    ax.plot(abs.(C[50,50:(end-50)])) #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
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
    ax.plot((-(bz*pi):((2*bz*pi)/m):(bz*pi) )./pi, real(mom_d1), "--o", markersize = 3., color = "tab:blue", label = "0.1")
    ax.plot((-(bz*pi):((2*bz*pi)/m):(bz*pi) )./pi, real(mom_d2), "--o", markersize = 3., color = "tab:red", label = "0.4")
    ax.plot((-(bz*pi):((2*bz*pi)/m):(bz*pi) )./pi, real(mom_d3), "--o", markersize = 3., color = "tab:green", label = "0.8")
    #ax.set_title("mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)"); #ax[1].grid(true)
    plt.legend(title = L"J_2/J_1")
    ax.set_ylabel("Momentum distribution"); #ax[1].grid(true)
    #ax.set_xscale()
    ax.set_xlabel("k"); #ax[3].grid(true)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%g π"))
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(base=0.5))
    plt.savefig("comp_mom_U($U)_N($N)_J2($J2)_dim($loc_hilb).pdf")

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
    plt.savefig("comp_C_U($U)_N($N)_J2($J2)_dim($loc_hilb).pdf")
    plt.show()
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
    plt.savefig("comp_Cabs_U($U)_N($N)_J2(0.1)_dim($loc_hilb).pdf")

    fig1, ax1 = plt.subplots(1,1, figsize=(4,4), tight_layout = true, dpi = 200)
    ax1.plot(abs.(real.(C2[50,50:(end-50)])), label = "Real part", color = "tab:green") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax1.plot(imag.(C2[50,50:(end-50)]), label = "Imag. part", color = "tab:orange") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    #ax1.plot(abs.(C2[50,50:(end-50)]), label = "Abs. value", color = "tab:red") #.* sin.(pi*(0:150) ./(length(C[:,1])-100)) .^2)
    plt.legend()
    ax1.set_title("CSF")
    ax1.set_ylabel("Correlation function"); #ax[1].grid(true)
    ax1.set_xlabel("|i-j|"); #ax[3].grid(true)
    ax1.set_xlim(-1,100)
    plt.savefig("comp_temp_U($U)_N($N)_J2(0.5)_dim($loc_hilb).pdf")
end

for J2 in [0.2, 0.225, 0.25, 0.275, 0.3]
    U = 0; loc_hilb = 4; N = 200; reupload = false; #J2 = 0.1
    main_dmrg(J2,U,N,loc_hilb, reupload = reupload)
    main_obs(J2,U,N,loc_hilb)
    #plot_mom_dist(J2,U,N,loc_hilb, 200, 1.)
    #plot_mom_dist_4(J2,U,N,loc_hilb, 1.)
end


function momentum_distribution_fft(J2,U,N,loc_hilb) #fourier transfor of hopping
    f = h5open("MPS.h5", "r")
    psi = read(f,"psi_U($U)_N($N)_J2($J2)_dim($loc_hilb)", MPS)
    close(f)   
    
    start = 50; endd = 200
    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix
    plt.figure(1)
    plt.plot(C[start,start:endd])
    plt.plot(cos.(-0.283) .* 0:50)

    ft = 0.0 + 0im; i = 70
    t0 = start - i              # Start time 
    tmax = endd - i       # End time       
    fs = tmax - t0          # Sampling rate (Hz)
    signal = C[start:endd,i]

    F = fftshift(fft(signal))
    freqs = fftshift(fftfreq(length(signal), fs))

    fig, ax = plt.subplots(1,1, figsize=(6,6), tight_layout = true, dpi = 150)
    ax.plot(freqs, F, "--o", color = "tab:blue")
    ax.set_title("mom_U($U)_N($N)_J2($J2)_dim($loc_hilb)"); #ax[1].grid(true)
    ax.set_ylabel(L"Momentum distribution"); #ax[1].grid(true)
    ax.set_xlabel("k"); #ax[3].grid(true)
end

#=
plot_mom_dist(J2,U,N,loc_hilb, 200, 1.)
plot_mom_dist_2(J2,U,N,loc_hilb, 1.)
=#

#=
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



