#PYPLOT options (took a while to figure out so DO NOT CHANGE)
using PyCall; ENV["KMP_DUPLICATE_LIB_OK"] = true
import PyPlot
const plt = PyPlot; 
plt.matplotlib.use("TkAgg"); ENV["MPLBACKEND"] = "TkAgg"; plt.pygui(true); plt.ion()

using LinearAlgebra, SparseArrays, Combinatorics, ITensors, ITensors.HDF5, MKL

function currents_from_correlation_V2(t, lattice::Vector{LatticeBond}, C, α=0)
    curr_plot = zeros(ComplexF64,length(lattice))
    couples = Any[]
    for (ind,b) in enumerate(lattice)
        push!(couples, [(b.x1, b.y1), (b.x2, b.y2)])
        curr_plot[ind] += 1im*t*(b.x2-b.x1)*C[b.s2, b.s1]-1im*t*(b.x2-b.x1)*C[b.s1, b.s2]
        curr_plot[ind] += -1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s1,b.s2]
        curr_plot[ind] += 1im*t*(b.y2-b.y1)*exp(+2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s2,b.s1]
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


function main(Nx, Ny; μ = 0.0, t = 1.0, U = 5.0, α = 1//4, ν = 1, #Nx is the easy dimension, Ny the difficult
                sparse_multithreading = false)
    
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
    Nϕ = α*(Nx-1)*(Ny-1) #number of fluxes
    loc_hilb = 4 #local hilbert space
    @show num_bos = ν * Nϕ 
    sites = siteinds("Boson", N; dim=loc_hilb, conserve_qns=true)
    lattice = square_lattice(Nx, Ny; yperiodic=false)

    #=set up of DMRG schedule=#
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, 100, 200, 300, 400, 500, 600, 700)
    setcutoff!(sweeps, 1e-8)
    setnoise!(sweeps, 1e-3, 1e-5, 1e-7, 1e-9, 0)
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

    if sparse_multithreading
        ITensors.disable_threaded_blocksparse()
        ITensors.Strided.set_num_threads(16)
        BLAS.set_num_threads(16)
    end

    #= write mps to disk =#
    f = h5open("MPS_HDF5/prova.h5","w")
    write(f,"Nx=($Nx)_Ny=($Ny)",psi)
    close(f)

    #= observables computation =#
    H2 = inner(H, psi, H, psi)
    variance = H2-energy^2
    @show real(variance) #variance to check for convergence

    ent_entr = zeros(N-3)
    for (ind, j) in enumerate((1:1:N)[2:end-2])
        ent_entr[ind] = entanglement_entropy(psi, j) #entanglemet entropy
    end

    C = correlation_matrix(psi, "Adag", "A") #correlation matrix
    dens = expect(psi, "N") #density
    @show sum(dens)
    C2, C3 = local_correlations(N,sites) #local correlation operators
    exp_C2 = @show real(inner(psi,C2,psi)/num_bos)
    exp_C3 = @show real(inner(psi,C3,psi)/num_bos)
    couples, curr_plot = currents_from_correlation_V2(t, lattice, C, α) #current

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

main(17,6, sparse_multithreading=true) #worth setting sparse_multithreading to false if Ny <4

#study of time scaling (using 17 Nx and increasing Ny, keeping always the same sweep schedule)
#trunc error 10e-9, 14 sweeps, alpha = 1/4
#1 1.25
#2 10.75 linkdim 9 
#3 31.19 linkdim 34
#4 71.36 linkdim 100
#5 158.30 linkdim 301
#6 397.96 saturation 450
#7 740.526 
