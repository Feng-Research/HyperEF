# Iterative Non-sampling method summing up all the vector values (applying QR)
# generating 5 vectors for each smoothing step

# multi level hypergraph decompositioning only using HyperEF

# This is the clean version of HyperEF_new

using TickTock
using SparseArrays
using LinearAlgebra
using Clustering
using NearestNeighbors
using Distances
using Metis
using Laplacians
using Arpack
using Plots
using Statistics
using DelimitedFiles
using StatsBase
using Random
using GraphPlot

include("hmet2ar.jl")
include("Star.jl")
include("Filter.jl")
include("h_score3.jl")
include("h_score.jl")
include("mx_func.jl")
include("Whgr_weighted.jl")
include("Whgr.jl")
include("INC3.jl")
include("StarW.jl")
include("Filter_new.jl")
include("Filter_new2.jl")
include("Clique.jl")
include("Rmtx.jl")
include("Clique_V2.jl")
include("HyperNodes.jl")
include("Helper_Functions.jl")
include("Filter_fast.jl")
include("mtx2ar.jl")



#filename = ["ibm01.hgr", "ibm02.hgr", "ibm03.hgr", "ibm04.hgr", "ibm05.hgr","ibm06.hgr", "ibm07.hgr", "ibm08.hgr", "ibm09.hgr", "ibm10.hgr", "ibm11.hgr", "ibm12.hgr", "ibm13.hgr", "ibm14.hgr", "ibm15.hgr", "ibm16.hgr", "ibm17.hgr", "ibm18.hgr"]

filename = ["ibm01.hgr"]

MNC = zeros(Float64, 0)
Pvec = zeros(Int32, 0)
NRV = zeros(Int32, 0)
TM = zeros(Float64, length(filename))

for Bloop = 1:1#length(filename)

    ar = hmet2ar(filename[Bloop])
    #M = Rmtx("football.mtx")
    #ar = mtx2ar("Gmat_mesh2d.mtx")
    #ar = mtx2ar("football.mtx")
    #ar = [[1,2,3,4,5], [2,3,4,5,6,7,8], [4,7,8,9], [9,10], [2,3,4,5]]

    #ar = [[1,2,3,4], [3,6,7,8], [4,5,6],[8,9]]

    ar_org = copy(ar)

    global ar_new = Any[]

    mx = mx_func(ar)

    LN_org = length(ar_org)

    mx_org = mx_func(ar)

    global idx_mat = Any[]

    Neff = zeros(Float64, mx)

    global W = ones(Float64, length(ar))

    println("---------------------new test case----------------------------")

    #while mx_eff < delta
    TM[Bloop] = @elapsed @inbounds for loop = 1:1

        println("Lar = ", length(ar))

        global W = ones(Float64, length(ar))

        mx = mx_func(ar)

        ## star expansion
        A = Star(ar)

        ## computing the smoothed vectors
        initial = 0

        SmS = 300

        interval = 20

        Nrv = 1

        RedR = .4

        Nsm = Int((SmS - initial) / interval)

        Ntot = Nrv * Nsm

        Qvec = zeros(Float64, 0)

        global Eratio = zeros(Float64, length(ar), Ntot)

        SV = zeros(Float64, mx, Ntot)

        for ii = 1:Nrv

            sm = zeros(mx, Nsm)

            Random.seed!(1); randstring()

            rv = (rand(Float64, size(A, 1), 1) .- 0.5).*2

            sm = Filter_fast(rv, SmS, A, mx, initial, interval, Nsm)

            SV[:, (ii-1)*Nsm+1 : ii*Nsm] = sm

        end

        ## Make all the smoothed vectors orthogonal to each other
        QR = qr(SV)

        SV = Matrix(QR.Q)

        ## Computing the ratios using all the smoothed vectors
        for jj = 1:size(SV, 2)

            hscore = h_score3(ar, SV[:, jj])

            Eratio[:, jj] = hscore ./ sum(hscore)

        end #for jj

        ## Approximating the effective resistance of hyperedges by selecting the top ratio
        #global Evec = sum(Eratio, dims=2) ./ size(SV,2)
        E2 = sort(Eratio, dims=2, rev=true)
        global Evec = E2[:, 1]

        # Adding the effective resistance of super nodes from previous levels
        @inbounds for kk = 1:length(ar)

            nd2 = ar[kk]

            Evec[kk] = Evec[kk] + sum(Neff[nd2])

        end

        println("maximum eff = ", maximum(Evec))

        ## Normalizing the ERs
        P = Evec ./ maximum(Evec)

        ## Choosing a ratio of all the hyperedges
        Nsample = round(Int, RedR * length(ar))

        global PosP = sortperm(P[:,1])

        ## Increasing the weight of the hyperedges with small ERs
        W[PosP[1:Nsample]] = W[PosP[1:Nsample]] .* (1 .+  1 ./ P[PosP[1:Nsample]])

        ## Selecting the hyperedges with higher weights for contraction
        global Pos = sortperm(W, rev=true)
        #global Pos = [4,3,2,1]

        ## Hyperedge contraction
        flag = falses(mx)

        flagE = falses(length(ar))

        val = 1

        global idx = zeros(Int, mx)

        Hcard = zeros(Int, 0)

        Neff_new = zeros(Float64, 0)

        @inbounds for ii = 1:Nsample

            nd = ar[Pos[ii]]

            fg = flag[nd]

            fd1 = findall(x->x==0, fg)

            if length(fd1) > 1

                nd = nd[fd1]

                flagE[Pos[ii]] = 1

                idx[nd] .= val

                flag[nd] .= 1

                append!(Hcard, length(ar[ii]))

                val +=1

                ## creating the super node weights
                new_val = Evec[Pos[ii]] + sum(Neff[nd])

                append!(Neff_new, new_val)

            end # endof if

        end #end of for ii

        ## indexing the isolated nodes
        fdz = findall(x-> x==0, idx)

        fdnz = findall(x-> x!=0, idx)

        global V = vec(val:val+length(fdz)-1)

        idx[fdz] = V

        #idx = [4,3,2,3,2,0,1,2,1,0].+1
        #idx = [3,1,0,3,2,2,1,0,1].+1

        ## Adding the weight od isolated nodes
        append!(Neff_new, Neff[fdz])

        push!(idx_mat, idx)

        ## generating the coarse hypergraph
        ar_new = Any[]

        @inbounds for ii = 1:length(ar)

            nd = ar[ii]

            nd_new = unique(idx[nd])

            push!(ar_new, sort(nd_new))

        end #end of for ii

        ## Keeping the edge weights of non unique elements
        #fdnu = unique(z -> ar_new[z], 1:length(ar_new))
        #W2 = W[fdnu]


        ## removing the repeated hyperedges
        global ar_new = unique(ar_new)

        ### removing hyperedges with cardinality of 1
        HH = INC3(ar_new)
        ss = sum(HH, dims=2)
        fd1 = findall(x->x==1, ss[:,1])
        deleteat!(ar_new, fd1)


        ar = ar_new

        Neff = Neff_new


    end #end for loop

    #println("ar_new = ", ar_new)


    ## Mapper

    global idx1 = 1:maximum(idx_mat[end])

    @inbounds for ii = length(idx_mat):-1:1

        idx1 = idx1[idx_mat[ii]]

    end # for ii


    ## global conductance

    ar = ar_org

    NH = HyperNodes(ar)

    H = INC3(ar)

    order = vec(round.(Int, sum(H, dims = 2)))

    vol_V = vec(round.(Int, sum(H, dims = 1)))

    global Cvec = zeros(Float64, 0)


    dict = Dict{Any, Any}()

    count = 0
    @inbounds for jj =1:length(idx1)

        vals = get!(Vector{Int}, dict, idx1[jj])

        push!(vals, jj)

    end # for jj

    KS = collect(keys(dict))

    global VL = collect(values(dict))

    global szCL_HE = zeros(Int, length(VL))

    ndV = collect(1:mx_func(ar))

    @inbounds for ii = 1:length(KS)

        S = VL[ii]

        szCL_HE[ii] = length(S)

        ct = tl_cut(H, vec(S), 1.0, order)

        vol1 = sum(vol_V[S])

        S_hat = deleteat!(ndV, S)

        vol2 = sum(vol_V[S_hat])

        cnd = ct / vol1

        push!(Cvec, cnd)

        ndV = collect(1:mx_func(ar))


    end # for ii


    ### Computing unbalance factor
    dict4 = Dict{Any, Any}()

    count = 0
    @inbounds for jj =1:length(szCL_HE)

        vals = get!(Vector{Int}, dict4, szCL_HE[jj])

        push!(vals, jj)

    end # for jj

    KS4 = collect(keys(dict4))

    VL4 = collect(values(dict4))

    sm = 0

    @inbounds for ii = 1:length(KS4)

        sm += KS4[ii] * length(VL4[ii])

    end

    UB = sm / length(szCL_HE)

    ########


    println("Unblanace factor = ", maximum(szCL_HE) / mean(szCL_HE))


    println("V_new: ", mx_func(ar_new))

    println("E_new: ", length(ar_new))

    NR = (mx_org - mx_func(ar_new)) / mx_org *100

    ER = (LN_org - length(ar_new)) / LN_org *100

    println("NR = ", NR)

    println("ER = ", ER)

    println("average conductance: ", round(mean(Cvec), digits=2))

    println("# Partitions: ", maximum(idx1))

    global MNC, Pvec, NR

    push!(MNC, round(mean(Cvec), digits=2))
    push!(Pvec, maximum(idx1))
    push!(NRV, round(Int,NR))

    Whgr("ibm01_re1.hgr", ar_new)



    #=
    ## coloring graph
    #ar = mtx2ar("karate.mtx")
    #ar = [[1,2,3,4,5], [2,3,4,5,6,7,8], [4,7,8,9], [9,10], [2,3,4,5]]
    ar = [[1,2,3,4], [3,6,7,8], [4,5,6],[8,9]]


    M = Star(ar)

    #vec1 = maximum(idx1)+1:size(M,1)
    append!(idx1, [11,11,11,11])

    fdnz = findnz(triu(M,1))

    rr = fdnz[1]
    cc = fdnz[2]
    vv = ones(Int, length(rr))

    using Graphs, SimpleWeightedGraphs
    g = SimpleWeightedGraph(rr,cc,vv)

    #gplot(g)

    nodecolor = range(colorant"blue", stop=colorant"red", length=maximum(idx1))


    nodefillc = nodecolor[idx1]

    #nodelabel = [1:num_vertices(g)]

    gplot(g, nodefillc=nodefillc, nodelabel=1:13)
    #gplot(g, nodefillc=nodefillc)

    =#




    #=
    ## Hypergraph coloring
    MM = Star(ar_org)
    L = lap(MM)

    EV = eigs(L; nev=3, which =:SM)

    EVs = EV[2]

    ev2 = EVs[:,2]

    ev3 = EVs[:,3]

    fd1 = VL[1]

    e2 = ev2[fd1]

    e3 = ev3[fd1]

    #plot(ev2, ev3, seriestype = :scatter)

    PP = plot(e2, e3, seriestype = :scatter, markersize=1,legend=false, markercolor=RGBA(rand(1)[1], rand(1)[1], rand(1)[1], rand(1)[1]))

    PP

    for ii = 2:length(VL)

        fd1 = VL[ii] #findall(x->x==ii, V)

        e2 = ev2[fd1]

        e3 = ev3[fd1]

        PP = plot!(e2, e3, seriestype = :scatter, legend=false, markercolor=RGBA(rand(1)[1], rand(1)[1], rand(1)[1], rand(1)[1]))


    end

    PP
    =#

end #for Bloop



#=
## Computing the real effective resistances

#ar = hmet2ar("ibm01.hgr")
ar = mtx2ar("football.mtx")

include("Clique_sm.jl")
M = Clique_sm(ar)

L = lap(M)

L[1,1] = L[1,1] .+ .1



R = zeros(Float64, 0)

R_star = zeros(Float64, 0)

for ii = 1:length(ar)

    global R_star

    node = ar[ii]

    N1 = node[1]

    N2 = node[2]

    u = zeros(Int, size(L, 1))

    u[N1] = 1

    u[N2] = -1

    V = L \ u

    r = abs(V[N1] - V[N2])

    append!(R_star, r)


end #end of ii

PP = plot(R_star, Evec, seriestype = :scatter, legend=false, fmt=:png)
xlabel!("Actual Effective resistance")
ylabel!("Approximated Effective Resistance")

savefig(PP, "cor_fig")
=#


#=
##hmetis
ar = hmet2ar("ibm05.hgr")
H = INC3(ar)
include("CUT.jl")

ss = sum(H, dims=1)

P2 = zeros(Int, 0)
open("ibm05_t1.hgr.part.128") do io

    while !eof(io)

        x = readline(io)

        xint = parse(Int, x)

        append!(P2, xint)

    end

end

P2 .= P2.+1


for ii = 1:length(idx_mat)

    idx_new = idx_mat[end - ii + 1]

    P2 = P2[idx_new]

end


mx_P = maximum(P2)
final_cond = zeros(Float64, 0)

ct_mul = 0

for ii = 1:mx_P

    nd = findall(x->x==ii, P2)

    nd_hat = findall(x->x!=ii, P2)

    vol1 = sum(ss[nd])

    vol2 = sum(ss[nd_hat])

    parts = ones(Int, length(P2))

    parts[nd] .= 2

    global ct = CUT(ar, parts)

    global ct_mul += ct

    CC = ct / min(vol1, vol2)

    append!(final_cond, CC)

end #end of ii

ct_final = CUT(ar, P2)

println("Hmetis Cut = ", ct_final)

println("Hmetis Conductance = ", maximum(final_cond))


=#
