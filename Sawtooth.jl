#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, SparseArrays, Combinatorics, ITensors, ITensors.HDF5, MKL, DelimitedFiles, ITensorGLMakie, JLD

function ITensors.op(::OpName"Sz", ::SiteType"Boson", s1::Index)
    sz = (1/4)*op("Id", s1) - op("N", s1)
    return sz
end

function ITensors.op(::OpName"expHOP", ::SiteType"Boson", s1::Index, s2::Index; τ, J)
    h =
        -J * op("Adag", s1) * op("A", s2) +
        -J * op("Adag", s2) * op("A", s1)
    return exp(τ * h)
end

function ITensors.op(::OpName"expU", ::SiteType"Boson", n::Index; τ, U)
    h =
        U/2 *prime(prime(op("N", n))* op("N", n),-1,2) - U/2 * op("N", n) 
    return exp(τ * h)
end

function tebd(psi; cutoff=1E-9, δt=0.05, ttotal=1., J1=-sqrt(2), J2=-1, U=0)
    s=siteinds(psi)
    N=length(psi)

    # Compute the number of steps to do
    Nsteps = Int(ttotal / δt)

    # Make gates (1,2),(2,3),(3,4),...
    gates_NN_odd = ops([("expHOP", (n, n + 1), (τ=-δt * im / 2, J=J1*(-1)^(n-1))) for n in 1:2:(N - 1)], s)
    gates_NN_even = ops([("expHOP", (n, n + 1), (τ=-δt * im / 2, J=J1*(-1)^(n-1))) for n in 2:2:(N - 1)], s)

    gates_NNN_odd = ops([("expHOP", (n, n + 2), (τ=-δt * im / 2, J=J2)) for n in 1:4:(N - 2)], s)
    gates_NNN_even = ops([("expHOP", (n, n + 2), (τ=-δt * im / 2, J=J2)) for n in 3:4:(N - 2)], s)

    gates_U = ops([("expU", n, (τ=-δt * im , U=U)) for n in 1:N], s)

    gates = ITensor[]
    append!(gates, gates_U)
    append!(gates, gates_NN_odd)
    append!(gates, gates_NNN_odd)
    append!(gates, gates_NN_even)
    append!(gates, gates_NNN_even)

    append!(gates, gates_NNN_even)
    append!(gates, gates_NN_even)
    append!(gates, gates_NNN_odd)
    append!(gates, gates_NN_odd)

    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...

    # Initialize psi to be a product state (alternating up and down)

    # Compute and print initial <Sz> value
    t = 0.0
    for step in 1:Nsteps
        psi = apply(gates, psi; cutoff)
        t += δt
    end
    return psi
end

function dmrg_run(N, J1, J2, J3, U; loc_hilb=4, sparse_multithreading = false, in_state = nothing)
    
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
    sweeps = Sweeps(200)
    setmaxdim!(sweeps, 200, 200, 200, 200, 300, 300, 300, 400, 400, 500, 600)
    setcutoff!(sweeps, 1e-9)
    setnoise!(sweeps, 0)
    en_obs = DMRGObserver(energy_tol = 1e-10)

    #in_state = [n%4==0 ? "1" : "0" for n in 1:N] #half filling
    #in_state[21] = "1"; #in_state[31] = "1"; in_state[35]="1"
    psi0 = productMPS(sites, in_state)

    #= hamiltonian definition =#
    ampo = OpSum()  
    for b1 in 1:(N-1) #+1: no frustration
        ampo += -J1*(-1)^(b1-1), "Adag", b1, "A", (b1+1)
        ampo += -J1*(-1)^(b1-1), "Adag", (b1+1), "A", b1
    end
    for b1 in 1:2:(N-2) #sawtooth: only odd sites have NNN hopping
        ampo += -J2, "Adag", b1, "A", (b1+2)
        ampo += -J2, "Adag", (b1+2), "A", b1
    end
    for b1 in 2:2:(N-2) #second rung hopping
        ampo += -J3, "Adag", b1, "A", (b1+2)
        ampo += -J3, "Adag", (b1+2), "A", b1
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
    return psi
end

function structure_factor(k, C)
    ft = 0
    for (a,i) in enumerate(1:2:length(diag(C)))
        for (b,j) in enumerate(1:2:length(diag(C)))
            ft += (C[j, i])*exp(1im*k*(b-a))
        end
    end
    return ft/length(psi)
end

C = correlation_matrix(psi,"Adag","A")
plt.figure(1,dpi=100)
plt.plot(1:length(psi),diag(C))
plt.scatter(1:length(psi),diag(C))
plt.title("N=40, T=0 J1/J2=sqrt(2), U=$U, in.state=N/4")
plt.xlabel("Position")
plt.ylabel("density")
plt.grid(true)
plt.savefig("T0-U$U-N4.pdf")


Cd = correlation_matrix(psi,"Sz","Sz")
dens = diag(C)
str = Float64[]
for k in range(0,2pi,length=Int(N))
    push!(str, real(structure_factor(k, Cd)))
end
plt.figure(2,dpi=100)
plt.plot(range(0,2pi,length=Int(N)),str)


o_cdw = sum(dens[13:4:65] .- dens[15:4:67])/14
open("CDW_U($U)_N(80)_dim(2).txt", "a") do io
    writedlm(io, [real(J3) real(o_cdw)])
end


matr = readdlm("CDW_U($U)_N(80)_dim(2).txt", '\t')
plt.scatter(matr[1:end,1],abs.(matr[1:end,2]))
plt.grid(); plt.xlabel("J3/J2")
plt.ylabel("o_cdw")
plt.title("N=80, hardcore, J1/J2=sqrt(2)")

let
    N = 24; J1 = -sqrt(2); J2 = 1.; U = 1; J3 = 0.0;
    in_state = [(n)%4==0 ? "0" : "0" for n in 1:N] #half filling
    for i in 3:4:length(in_state)
        in_state[i] = "1"
    end

    #in_state[11] = "0"; #in_state[27] = "1"; in_state[27] = "1";
    psi = dmrg_run(N, J1, J2, J3, U, in_state = in_state)
    C = correlation_matrix(psi,"Adag","A")
    plt.scatter(1:length(psi),diag(C))
    plt.plot(1:length(psi),diag(C))
    time_ev = zeros(N,length(0.5:0.5:15)+1)
    time_ev[1:end,1] = real(diag(C))
    for (i, ttotal) in enumerate(0.5:0.5:15)
        psi=tebd(psi, ttotal=0.5, J2=(J2+0.2), J1=J1, U = U)
        C=correlation_matrix(psi,"Adag","A")
        time_ev[1:end, i+1] = real(diag(C))
        println(ttotal)
        println(maxlinkdim(psi))
        #=
        plt.figure(ttotal,dpi=100)
        plt.plot(1:length(psi),diag(C),linestyle="-")
        plt.scatter(1:length(psi),diag(C),linestyle="-")
        plt.title("N=40, T=$ttotal J1/J2=sqrt(2), U=$U, in.state=N/4+1")
        plt.xlabel("Position")
        plt.ylabel("density")
        plt.grid(true)
        =#
        #plt.savefig("plots/T$ttotal-U$U-N4+1.pdf")
    end

    fig, ax = plt.subplots(1,dpi=100)
    plt.title("N=24, T=15*J2 J1/J2=sqrt(2), U=$U, in.state=N/4")
    ax.set_xlabel("Time")
    ax.set_ylabel("position")
    sh = ax.imshow(time_ev, extent = [0,15,24,1], aspect = 0.4)
    plt.colorbar(sh, ax=ax, label="density")
    plt.tight_layout()
    plt.savefig("time_ev_n4_$U.pdf")    

    h5open("data2.h5","cw") do f
        if haskey(f,"time_ev_J202_$U")
            delete_object(f,"time_ev_J202_$U")
        end
        write(f,"time_ev_J202_$U",time_ev)
    end
end
U=5
let

    h5open("data2.h5","r") do f
        time_ev = read(f,"time_ev_n4-1_$U")
        fig, ax = plt.subplots(1,dpi=100)
        #plt.title("N=24, T=15*J2 J1/J2=sqrt(2), U=$U, in.state=N/4+1")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        sh = ax.imshow(time_ev, extent = [0,15,24,1], aspect = 0.6)
        plt.colorbar(sh, ax=ax, label="Density", pad=0.04)
        plt.tight_layout()
        #plt.savefig("time_ev_$U.pdf")    
    end

end
