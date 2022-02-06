using LinearAlgebra, SparseArrays, Plots, Arpack, Combinatorics, ITensors, ITensorGPU


function tight_binding(N,t,μ,A,BC="O")
    bst=10e-6
    mat=diagm(0 => fill(-μ-2t,N), 1 => fill(t*exp(1im*A+1im*bst),N-1), -1 => fill(t*exp(-1im*A-1im*bst),N-1))
    if BC=="P"   
        mat[1,end]=t*exp(-1im*A-1im*bst)
        mat[end,1]=t*exp(1im*A+1im*bst)
    end
    return sparse(mat)
end

function crni(ψ,ham,dt)
    ψₙ=(((I-1im*ham*dt/2)*ψ)\(I+1im*ham*dt/2))'
    return ψₙ
end

function lanczos(ψ,ham,nmax,dt)
    N=√(ψ⋅ψ)
    q=zeros(ComplexF64,length(ψ),nmax+1); q[:,1]=ψ/N
    beta=zeros(ComplexF64,nmax+1,nmax+1)
    for k in range(nmax+1)  
        temp=ham*q[:,k]
        for j in range(k+1)
            beta[j,k]=q[:,j]⋅temp
            temp-=q[:,j]*beta[j,k]
        end
        Nt=√(temp⋅temp)
        if k!=nmax
            beta[k+1,k]=Nt
            q[:,k+1]=temp/Nt
        end
    end
    lam,S=eigen(beta)
    c=S*diagm(exp(-1im*lam*dt))*S'[:,0]*N
    return q*c
end

##
N=3
ham=tight_binding(N,1,0,0)
vals,vecs=eigs(ham)
GS=vecs[:,1]
plot(abs.(GS).^2)

ham1=tight_binding(N,20,0,0)
dt=0.01
timeax = range(0,10,step=dt)
asc= collect(1:N)
dens=zeros(Float64,length(timeax))
for (i,t) in enumerate(timeax)
    println(i)
    GS=crni(GS,ham1,dt)
    dens[i]=real(dot(GS,diagm(asc),GS))
end

plot(timeax,dens.-((N+1)/2))


function base_construction(num_bos,num_sites)
    base_matr=zeros(Int64,num_sites+1,num_bos+1) #builds matrix for labeling of states
    for i in range(0,num_sites)
        for j in range(0,num_bos)
            base_matr[i+1,j+1]=binomial(i+j-1,j)
        end
    end
    avail_sites=range(1,num_sites)
    list_pos=[reverse(i) for i in collect(with_replacement_combinations(avail_sites,num_bos))]
    num_pos=fill(Int[],length(list_pos))
    for i in range(1,length(list_pos))
        num_pos[i]=[j-'0' for j in collect(lpad(string(sum((num_bos+1).^list_pos[i]), base=num_bos+1)[1:end-1],num_sites,'0'))]
    end
    labels=Dict()
    for i in range(1,length(list_pos))
        nb=1
        for j in range(1,num_bos)
            nb+=base_matr[num_sites-list_pos[i][j]+1,j+1]
        end
        labels[num_pos[i]]=Integer(nb)
    end
    return labels
end

function hopping_bhubbard(labels)
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
            # for k in range(j+1,len(out_state)):
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

function repulsion(labels)
    pot_en=Vector{ComplexF64}()
    for i in range(1,length(labels))    
        in_state=collect(keys(labels))[i]
        pot=0.5sum([a*b for (a,b) in zip(in_state,[x-1 for x in in_state])])
        push!(pot_en,pot) 
    end      
    return sparse(collect(values(labels)),collect(values(labels)),pot_en)
end

lab=base_construction(3,3)
hopping_bhubbard(lab)
repulsion(lab)

#= PARAMETERS DEFINITION =#
t=1; μ=0; U=0; W=0 #Hamiltonian parameters
T=6; dt=0.01 #Time evolution parameters
num_bos=4; num_sites=4

labels=base_construction(num_bos,num_sites)
ham=-t*hopping_bhubbard(labels)+U*repulsion(labels)-(μ-2t)*spdiagm(fill(num_bos,length(labels)))

#= EXACT DIAGONALIZATON =#
vals,vecs=eigs(ham)
GS=vecs[:,1]

# TIME EVOLUTION
initial_state=zeros(ComplexF64, length(labels))
initial_state[labels[fill(1,num_sites)]]=1
ψ=copy(initial_state)
for τ in range(0,T,step=dt)
    println(τ)
    psi=crni(ψ,ham,dt)
end
=#

let
    gpu = cu
    N = 100
    sites = siteinds("S=1",N)
  
    ampo = OpSum()
    for j=1:N-1
      ampo += "Sz",j,"Sz",j+1
      ampo += 1/2,"S+",j,"S-",j+1
      ampo += 1/2,"S-",j,"S-",j
    end
    H = gpu(MPO(ampo,sites))
  
    psi0 = gpu(randomMPS(sites,10))
  
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 10,20,100,100,200)
    setcutoff!(sweeps, 1E-10)
  
    energy, psi = dmrg(H,psi0, sweeps)
    return
  end
  
