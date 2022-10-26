cd("/Users/aliaghdaei/Desktop/Hypergraph/HypergraphFlowClustering-master/Exp-Amazon/Partitioning/HyperEF_git/")

using Statistics
using SparseArrays
using Random
using LinearAlgebra

include("HyperEF.jl")
include("Functions.jl")

include("HyperNodes.jl")

filename = "ibm01.hgr"

ar = ReadInp(filename)

ar_org = copy(ar)

mx_org = mxF(ar_org)
LN_org = length(ar_org)
## L: the number of coarsening levels, e.g. 1, 2, 3, 4, ...
L = 1

## R: Effective resistance threshold tor growing the clusters (0<R<=1)
R = 1

ar_new, idx_mat = HyperEF(ar, L, R)



global idx1 = 1:maximum(idx_mat[end])

@inbounds for ii = length(idx_mat):-1:1

    global idx1 = idx1[idx_mat[ii]]

end # for ii


## global conductance
include("HyperNodes.jl")
include("INC3.jl")
include("Helper_Functions.jl")
ar = ar_org

NH = HyperNodes(ar)

H = INC(ar)

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

global ndV = collect(1:mxF(ar))

@inbounds for ii = 1:length(KS)

    S = VL[ii]

    szCL_HE[ii] = length(S)

    ct = tl_cut(H, vec(S), 1.0, order)

    vol1 = sum(vol_V[S])

    S_hat = deleteat!(ndV, S)

    vol2 = sum(vol_V[S_hat])

    cnd = ct / vol1

    push!(Cvec, cnd)

    global ndV = collect(1:mxF(ar))


end # for ii



println("V_new: ", mxF(ar_new))

println("E_new: ", length(ar_new))

NR = (mx_org - mxF(ar_new)) / mx_org *100

ER = (LN_org - length(ar_new)) / LN_org *100

println("NR = ", NR)

println("ER = ", ER)

println("average conductance: ", round(mean(Cvec), digits=2))

Whgr("HyperEF_out.hgr", ar_new)
