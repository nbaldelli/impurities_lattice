function NKron(sites,op1,op2,pos) 
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
