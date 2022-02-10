using LinearAlgebra, SparseArrays, Plots, Arpack, Combinatorics, ITensors, ITensorGPU, MKL

?function NKron(sites,op1,op2,pos) 
    #=
    Tensor product operator 
    returns tensor product of op1,op2 on sites pos,pos+1 and identity on remaining sites
    L:number of sites
    op1,op2: Pauli operators on neighboring sites
    pos: site to insert op1
    =#
    Op1 = op(op1,sites[pos])
    Op2 = op(op2,sites[pos])

    ide = Matrix(I,dim(sites[1]),dim(sites[1]))
    mat=1
    for j in range(1,(pos-1))
        mat = kron(mat,ide)
    end
    mat = kron(mat,Array(Op1,inds(Op1)[1],inds(Op1)[2]))
    mat = kron(mat,Array(Op2,inds(Op2)[1],inds(Op2)[2]))
    for j in range(pos+2,length(sites))
        mat = kron(mat,ide)
    end
    return mat
end
function square_lattice_arr(Nx::Int, Ny::Int; kwargs...)
    yperiodic = get(kwargs, :yperiodic, false)
    yperiodic = yperiodic && (Ny > 2)
    N = Nx * Ny
    Nbond = 2N - Ny + (yperiodic ? 0 : -Nx)
    latt = Vector(undef, Nbond)
    b = 0
    for n in 1:N
      x = div(n - 1, Ny) + 1
      y = mod(n - 1, Ny) + 1
      if x < Nx
        latt[b += 1] = [n n+Ny x y x+1 y]
      end
      if Ny > 1
        if y < Ny
          latt[b += 1] = [n n+1 x y x y+1]
        end
        if yperiodic && y == 1
          latt[b += 1] = [n n+Ny-1 x y x y+Ny]
        end
      end
    end
    return latt
  end
end
function hopping_bhubbard(labels,lattice)
    br_term=10e-6
    in_states_list=Vector{Int64}()
    out_states_list=Vector{Int64}()
    kin_en=Vector{ComplexF64}()
    for i in range(1,length(labels))
        in_state=collect(keys(labels))[i]
        in_indexes=[i for (i, e) in enumerate(in_state) if e != 0]
        for j in in_indexes
            bos_coeff_dist=√(in_state[j])
            out_state=zeros(length(in_state))
            out_state[1:end]=in_state[1:end]
            out_state[j]=out_state[j]-1
            k=(j)%length(in_state)+1
            out_state[k]+=1
            bos_coeff_constr=√(out_state[k])
            push!(in_states_list,labels[in_state])
            push!(out_states_list,labels[out_state])
            push!(kin_en,bos_coeff_dist*bos_coeff_constr*exp(1im*br_term))
            out_state[k]-=1
        end
    end
    matr=sparse(in_states_list,out_states_list,kin_en)
    return matr.+matr'
end
function currents_MPO(t, lattice::Vector{LatticeBond}, sites,  α = 0) #Currents for ladder geometry with flux (this is the full MPO)
    curr_lowleg = OpSum(); curr_highleg = OpSum(); curr_rung = OpSum()

    for b in lattice
        curr_lowleg += 1im*t*(b.x2-b.x1)*isodd(b.s1), "Adag", b.s2, "A", b.s1 #current on the bottom leg
        curr_lowleg += -1im*t*(b.x2-b.x1)*isodd(b.s1), "Adag", b.s1, "A", b.s2

        curr_highleg += 1im*t*(b.x2-b.x1)*iseven(b.s1), "Adag", b.s2, "A", b.s1 #current on the top leg
        curr_highleg += -1im*t*(b.x2-b.x1)*iseven(b.s1), "Adag", b.s1, "A", b.s2

        curr_rung += 1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*b.x1*(b.y2-b.y1)), "Adag", b.s1, "A", b.s2 #current on the rungs
        curr_rung += 1im*t*(b.y2-b.y1)*exp(+2pi*1im*α*b.x1*(b.y2-b.y1)), "Adag", b.s2, "A", b.s1
    end
    return MPO(curr_lowleg, sites), MPO(curr_highleg, sites), MPO(curr_rung, sites)
end

function currents_from_correlation(t, lattice::Vector{LatticeBond}, C, Nx, Ny,  α = 0) #Currents for ladder geometry with flux (this is the full MPO)
    curr_lowleg = zeros(ComplexF64,Nx); curr_highleg = zeros(ComplexF64,Nx); curr_rung = zeros(ComplexF64,Nx)
    for b in lattice
        curr_lowleg[Int(b.x1)] += 1im*t*(b.x2-b.x1)*isodd(b.s1)*C[b.s2, b.s1] #current on the bottom leg
        curr_lowleg[Int(b.x1)] += -1im*t*(b.x2-b.x1)*isodd(b.s1)*C[b.s1, b.s2]

        curr_highleg[Int(b.x1)] += 1im*t*(b.x2-b.x1)*iseven(b.s1)*C[b.s2, b.s1] #current on the top leg
        curr_highleg[Int(b.x1)] += -1im*t*(b.x2-b.x1)*iseven(b.s1)*C[b.s1, b.s2]

        curr_rung[Int(b.x1)] += 1im*t*(b.y2-b.y1)*exp(-2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s1,b.s2] #current on the rungs
        curr_rung[Int(b.x1)] += -1im*t*(b.y2-b.y1)*exp(+2pi*1im*α*b.x1*(b.y2-b.y1))*C[b.s2,b.s1]
    end
    return curr_lowleg, curr_highleg, curr_rung
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

labels=base_construction(num_bos,Nx * Ny)   
lattice = square_lattice_arr(Nx, Ny; yperiodic=false)

#let #scaling up of magnetic flux
##
    μ = 0.0; t = 1.0; U = 0.0; αtot = 1//4;
    Nx = 19; Ny = 5 #Nx is the easy dimension, Ny the difficult
    N = Nx * Ny
    
    ν = 1//2 #filling

    Nϕ = αtot*(Nx-1)*(Ny-1) #number of fluxes
    @show num_bos = ν * Nϕ 
    #num_bos = 2

    #check increasing number of bosons if i get convergence (10 or 14) 
    #check that it is gapless (with correlation functions bdagger b)
    # with hardcore bosons with density 0.5 you are in mott otherwise ALWAYS superfluid
    # with other number of local spaces you can have a mott-superfluid transition depending on U/t

    loc_hilb = 2 #local hilbert space
    sweeps = Sweeps(10)
    setmaxdim!(sweeps, 10, 10, 20, 20, 40, 80, 80, 160, 320, 450)
    setcutoff!(sweeps, 1e-10)
    setnoise!(sweeps, 1e-3, 1e-5, 1e-7, 1e-9, 0)
    @show sweeps

    sites = siteinds("Boson", N; dim=loc_hilb, conserve_qns=true)
    lattice = square_lattice(Nx, Ny; yperiodic=false)

    state = [n<=num_bos ? "1" : "0" for n in 1:(N-0)]
    append!(state,fill("0",0))
    psi0 = randomMPS(sites, state)
        
    for α in [1/4]

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

        energy, psi = @time dmrg(H, psi0, sweeps, verbose=false);

        ent_entr = zeros(N-3)
        for (ind, j) in enumerate((1:1:N)[2:end-2])
            ent_entr[ind] = entanglement_entropy(psi, j)
        end

        C = correlation_matrix(psi, "Adag", "A")
        #c = display(scatter(C[Int(Nx/2), Int(Nx//2):end]))
        jlow, jhigh, jrung = currents_from_correlation(t, lattice, C, Nx, Ny, α)
        curr=round.(real(vcat(transpose(jlow),transpose(jrung),transpose(jhigh))),digits=9)

        H2 = inner(H, psi, H, psi)
        E = inner(psi, H, psi)
        var = H2-E^2
        @show var
        
        aa = reshape(expect(psi, "N"), (Ny, Nx))
        @show sum(aa)
        #var = @show expect(psi,H*H) - energy^2

        psi0 = deepcopy(psi)

        a = plot(real(ent_entr))
        title!(a,"ent. entr., loc_hilb=$loc_hilb, U=$U, α=$α")
        display(a)

        p = heatmap([string(i) for i in 1:Nx], [string(j) for j in 1:Ny], aa, c = :heat)
        title!(p,"dens, loc_hilb=$loc_hilb, U=$U, α=$α")
        vline!(p, [i for i in 1:Nx], c=:black)
        hline!(p, [j for j in 1:Ny], c=:black)
        display(p)

        d = heatmap([string(i) for i in 1:Nx], [string(j) for j in 1:(Ny+1)], curr, c = :heat)
        title!(d,"current, loc_hilb=$loc_hilb, U=$U, α=$α")
        vline!(d, [i for i in 1:Nx], c=:black)
        hline!(d, [j for j in 1:Ny], c=:black)
        display(d)

    end

#end
