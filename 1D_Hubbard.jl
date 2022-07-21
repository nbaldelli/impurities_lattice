using LinearAlgebra, SparseArrays, Plots, Arpack, Combinatorics, DataStructures

function tight_binding(N,t,μ,A,BC="O") #hopping hamiltonian (with EM field A peierls coupled)
    bst=10e-6
    mat=diagm(0 => fill(-μ-2t,N), 1 => fill(t*exp(1im*A+1im*bst),N-1), -1 => fill(t*exp(-1im*A-1im*bst),N-1))
    if BC=="P"   
        mat[1,end]=t*exp(-1im*A-1im*bst)
        mat[end,1]=t*exp(1im*A+1im*bst)
    end
    return sparse(mat)
end

function crni(ψ,ham,dt) #crank-nicholson evolution method
    ψₙ=(((I-1im*ham*dt/2)*ψ)\(I+1im*ham*dt/2))'
    return ψₙ
end

function lanczos(ψ,ham,nmax,dt) #lanczos evolution method
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
    labels=SortedDict()
    for i in range(1,length(list_pos))
        nb=1
        for j in range(1,num_bos)
            nb+=base_matr[num_sites-list_pos[i][j]+1,j+1]
        end
        labels[num_pos[i]]=Integer(nb)
    end
    return labels
end

function hopping_bhubbard(labels; dist=1, BC = "O", frustr = false)
    br_term=1e-18 #time reversal breaking (for pbc)
    in_states_list = Int64[]
    out_states_list = Int64[]
    kin_en = ComplexF64[]
    for i in range(1,length(labels))
        in_state=collect(keys(labels))[i] 
        in_indexes=[i for (i, e) in enumerate(in_state) if e != 0] #choose occupied sites
        for j in in_indexes
            if j >= (length(in_state)-dist+1) && BC == "O" 
                continue
            end
            print(in_state[j])
            bos_coeff_dist = √(in_state[j]) #coefficient related to bosonic destruction
            out_state = zeros(length(in_state))
            out_state[1:end] = in_state[1:end]
            out_state[j] = out_state[j] - 1 #destroy one particle 
            k = (j+dist-1)%(length(in_state)) + 1  #periodic boundary

            out_state[k] += 1
            bos_coeff_constr=√(out_state[k])

            push!(in_states_list,labels[in_state])
            push!(out_states_list,labels[out_state])
            frustr ? coef = -1 : coef = 1
            push!(kin_en,(coef^j)*bos_coeff_dist*bos_coeff_constr*exp(1im*br_term))
        end
    end
    matr=sparse(in_states_list,out_states_list,kin_en, length(labels),length(labels))
    return matr.+matr'
end

function repulsion(labels)
    pot_en = ComplexF64[]
    for i in range(1,length(labels))    
        in_state=collect(keys(labels))[i]
        pot=0.5sum([a*b for (a,b) in zip(in_state,[x-1 for x in in_state])])
        push!(pot_en,pot) 
    end      
    return sparse(collect(values(labels)),collect(values(labels)),pot_en)
end

#= PARAMETERS DEFINITION =#
let
    J1=-sqrt(2); J2=1.; μ=0; U=0; W=0 #Hamiltonian parameters
    T=2; dt=0.001 #Time evolution parameters
    num_sites=41; num_bos=1

    labels=base_construction(num_bos,num_sites)

    zigz = -J1*hopping_bhubbard(labels, dist=1, BC="O", frustr = true)
    rung = -J2*hopping_bhubbard(labels, dist=2, BC="O")
    hub = U*repulsion(labels)
    for i in 2:2:(num_sites-2)
        rung[i, i+2] = 0
        rung[i+2, i] = 0
    end

    ham=zigz+dropzeros(rung)+hub
    #-(μ-2t)*spdiagm(fill(num_bos,length(labels)))

    #= EXACT DIAGONALIZATON =#
    #vals,vecs=eigs(ham);
    vals, vecs = eigen(collect(ham))
    println(vals)
    GS = abs.(vecs[:,1].^2)

    dens = zeros(Float64, num_sites)
    for i in 1:length(labels)
        stated = collect(keys(labels))[i]
        dens .+= stated .* GS[get(labels,stated,0)]
    end
    #plt.scatter(1:num_sites,dens)

    # TIME EVOLUTION
    initial_state=zeros(ComplexF64, length(labels))
    in_conf = zeros(Int64,num_sites)
    in_conf[21] = 1; #in_conf[27] = 1
    initial_state[labels[in_conf]]=1
    ψ=copy(initial_state)
    
    for τ in range(0,T,step=dt)
        println(τ)
        ψ=crni(ψ,ham,dt)
    end
    
    dens = zeros(Float64, num_sites)
    GS = abs.(ψ .^2)
    for i in 1:length(labels)
        stated = collect(keys(labels))[i]
        dens .+= stated .* GS[get(labels,stated,0)]
    end
    plt.scatter(1:num_sites,dens)
    println(sum(dens))

end

plt.title("N=39, T=10 J1/J2=sqrt(2), in.state= 2 part.@13,27")
plt.xlabel("Position")
plt.ylabel("density")
plt.legend(["U=0","U=1","U=5","U=50"])
plt.grid(true)
