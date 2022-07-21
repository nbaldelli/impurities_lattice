#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, SparseArrays, Combinatorics, ITensors, ITensors.HDF5, MKL, DelimitedFiles

function currents_from_correlation_V2(t, lattice::Vector{LatticeBond}, C; α=0)
    curr_plot = zeros(ComplexF64,length(lattice))
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        println(ind)
        curr_plot[ind] += 1im*t*(b.x2-b.x1)*C[b.s2, b.s1]-1im*t*(b.x2-b.x1)*C[b.s1, b.s2]
        curr_plot[ind] += -1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s1,b.s2]
        curr_plot[ind] += 1im*t*(b.y2-b.y1)*exp(+2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s2,b.s1]
        println([(b.x1, b.y1), (b.x2, b.y2)])
        println(curr_plot[ind])
    end
    return couples, real(curr_plot)
end

function entanglement_entropy(psi,b)
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    p=0.0
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
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


function dmrg_run(Nx, Ny, μ, t, U, α, ν, #Nx is the easy dimension, Ny the difficult
                sparse_multithreading = false, yperiodic = true)
    
    if sparse_multithreading
        ITensors.Strided.set_num_threads(1)
        BLAS.set_num_threads(1)
        ITensors.enable_threaded_blocksparse()
    else
        ITensors.Strided.set_num_threads(16)
        BLAS.set_num_threads(16)
        ITensors.disable_threaded_blocksparse()
    end

    N = Nx * Ny
    Nϕ = yperiodic ? α*(Nx-1)*(Ny) : α*(Nx-1)*(Ny-1) #number of fluxes
    loc_hilb = 4 #local hilbert space
    @show num_bos = ν * Nϕ 
    
    sites = siteinds("Boson", N; dim=loc_hilb, conserve_qns=true)
    lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)

    #=set up of DMRG schedule=#
    sweeps = Sweeps(15)
    setmaxdim!(sweeps, 200, 200, 200, 200, 200, 300, 300, 400, 400,)
    setcutoff!(sweeps, 1e-8)
    setnoise!(sweeps,  1e-6, 1e-6, 0)
    #in_state = [n<=num_bos ? "1" : "0" for n in 1:(N-0)] #all bosons on the left, may be suboptimal
    in_state = [(n<=num_bos/2)||(n>(N-num_bos/2)) ? "1" : "0" for n in 1:(N-0)] #half on left half on right
    #in_state = [(n%(Ny-1)==0)&&(n<=num_bos*(Ny-1)) ? "1" : "0" for n in 1:(N-0)] #all bosons on the left, may be suboptimal
    append!(in_state,fill("0",0))
    psi0 = randomMPS(sites, in_state)

    #= hamiltonian definition =#
    ampo = OpSum()
    for b in lattice
        ampo += -t*exp(-2pi*1im*α*b.x1*(b.y2-b.y1)), "Adag", b.s1, "A", b.s2 #hopping with flux
        ampo += -t*exp(+2pi*1im*α*b.x1*(b.y2-b.y1)), "Adag", b.s2, "A", b.s1
    end
    for n in 1:N
        ampo += -μ, "N", n #chemical potential
        ampo += U, "N", n, "N", n #on-site interaction
        ampo += -U, "N", n 
    end

    H = MPO(ampo, sites)

    if sparse_multithreading
        H = splitblocks(linkinds,H)
    end
    ####################################
    #= DMRG =#
    energy, psi = @time dmrg(H, psi0, sweeps, verbose=false);
    ####################################

    #= write mps to disk =#
    f = h5open("MPS_HDF5/local.h5","w")
    write(f,"Nx_($Nx)_Ny_($Ny)_mu_($μ)_t_($t)_U_($U)_alpha_($α)_nu_($ν)",psi)
    write(f,"pr",psi)
    close(f)

    H2 = inner(H, psi, H, psi)
    variance = H2-energy^2
    @show real(variance) #variance to check for convergence

    return energy, variance, psi
end

function observable_computation(psi, Nx, Ny, μ, t, U, α, ν, yperiodic = true)

    ITensors.disable_threaded_blocksparse()
    ITensors.Strided.set_num_threads(16)
    BLAS.set_num_threads(16)

    
    if (psi === nothing)
        f = h5open("MPS_HDF5/local.h5","r")
        psi = read(f,"Nx_($Nx)_Ny_($Ny)_mu_($μ)_t_($t)_U_($U)_alpha_($α)_nu_($ν)",MPS)
        close(f)
    end
    
    loc_hilb = 4
    N = Nx * Ny
    #sites = siteinds(psi)
    lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)

    #= observables computation =#
    C = correlation_matrix(psi, "Adag", "A"); #correlation matrix
    dens=real(diag(C))
    #matr = readdlm("data_2D/dens_Nx_($Nx)_Ny_($Ny)_mu_($μ)_t_($t)_U_($U)_alpha_($α)_nu_($ν).txt", '\t')
    #dens=matr[1:end]

    #=
    C2, C3 = local_correlations(N,sites) #local correlation operators
    exp_C2 = @show real(inner(psi,C2,psi)/sum(dens))
    exp_C3 = @show real(inner(psi,C3,psi)/sum(dens))
    =#
    couples, curr_plot = currents_from_correlation_V2(t, lattice, C, α=α) #current
    println(curr_plot)
    
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
    end
    #matr = readdlm("data_2D/curr_Nx_($Nx)_Ny_($Ny)_mu_($μ)_t_($t)_U_($U)_alpha_($α)_nu_($ν).txt", '\t')
    #curr_plot=matr[1:end]

    #= plot of current and density =#
    fig, ax = plt.subplots(1, dpi = 220)
    ax.set_ylim(0.5,Ny+0.5)
    line_segments = plt.matplotlib.collections.LineCollection(couples,array=curr_plot, 
                                                            norm=plt.matplotlib.colors.Normalize(vmin=minimum(curr_plot), vmax=maximum(curr_plot)),
                                                            linewidths=5, cmap=plt.get_cmap("RdBu_r"), rasterized=true, zorder = 0)
    pl_curr = ax.add_collection(line_segments)
    pl_dens = ax.scatter(repeat(1:Nx, inner = Ny), repeat(1:Ny,Nx), c=dens, s=170, marker="s", zorder = 1, cmap=plt.get_cmap("PuBu"), edgecolors="black")
    plt.gca().set_aspect("equal")
    plt.colorbar(pl_dens, ax=ax, location="bottom", label="density", shrink=0.7, pad=0.03, aspect=50)
    plt.colorbar(pl_curr, ax=ax, location="bottom", label="current", shrink=0.7, pad=0.07, aspect=50)
    plt.title("Parameters: α=$α, Nx=$Nx, Ny=$Ny, ν=$ν, U=$U, loc_hilb=$loc_hilb")
    plt.tight_layout()
    display(fig)
    plt.close()
end

#= PARAMETERS =#
Nx = 5; Ny = 4; μ = 0.0; t = 1.0; U = 1.0; α = 1/6; ν = 1; sparse_multithreading = true; yperiodic = false

E, σ, GS = dmrg_run(Nx, Ny, μ, t, U, α, ν, sparse_multithreading, yperiodic) #worth setting sparse_multithreading to false if Ny <4
observable_computation(nothing, Nx, Ny, μ, t, U, α, ν, yperiodic)
