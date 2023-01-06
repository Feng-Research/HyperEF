## Input: input dataset in hMetis format
# Outout: hypergraph array
function ReadInp(input)

    io = open(input, "r")

    ar  = Any[]
    while !eof(io)
        rr = zeros(Int, 0)
        ln = readline(io)
        sp = split(ln)

        for kk = 1:length(sp)
            r = parse(Int, sp[kk])
            append!(rr, r)
        end #kk
        push!(ar, rr)

    end

    ar = deleteat!(ar, 1)

    return ar

end #end of function

## Input: hypergraph array
# Outout: sparse incidence matrix
function INC(ar)

    col = zeros(Int, 0)
    row = zeros(Int, 0)



    for iter = 1:length(ar)
        cc = (iter) * ones(Int, length(ar[iter]))
        rr = ar[iter]

        append!(col, cc)
        append!(row, rr)
    end

    row = row

    val = ones(Float64, length(row))

    mat = sparse(col, row, val)

    return mat
end

## Input: hypergraph array
# Output: number of nodes in hypergraph
function mxF(ar)

    mx2 = Int(0)
    aa = Int(0)

    for i =1:length(ar)

    	mx2 = max(aa, maximum(ar[i]))
    	aa = mx2

    end
    return mx2

end

## Input: hypergraph array, hyperedge weights
# Output: sparse simple graph
function StarW(ar, W)

    mx = mxF(ar)

    sz = length(ar)
    col = zeros(Int32, 0)
    val = zeros(Float32, 0)
    row = zeros(Int32, 0)

    for iter =1:length(ar)
        LN = length(ar[iter])
        cc = (iter+mx) * ones(Int, LN)
        vv = (W[iter]/LN) * ones(Int, LN)

        rr = ar[iter]
        append!(col, cc)

        append!(row, rr)

        append!(val, vv)
    end

    mat = sparse(row, col, val,mx+sz, mx+sz)

    A = mat + mat'

    return A

end


## Input: a set of random vectors, smoothing steps, star matrix, number of nodes in hypergraph
# index of the first selected smoothed vector, interval among the selected smoothed vectors, total number of smoothed vectors
# Output: a set of smoothed vectors
function Filter(rv, k, AD, mx, initial, interval, Ntot)

    sz = size(AD, 1)

    V = zeros(mx, Ntot);

    sm_vec = zeros(mx, k);

    AD = AD .* 1.0

    AD[diagind(AD, 0)] = AD[diagind(AD, 0)] .+ 0.1

    dg = sum(AD, dims = 1) .^ (-.5)

    I2 = 1:sz

    D = sparse(I2, I2, sparsevec(dg))

    on = ones(Int, length(rv))

    sm_ot = rv - ((dot(rv, on) / dot(on, on)) * on)

    sm = sm_ot ./ norm(sm_ot);

    count = 1

    for loop in 1:k

        sm = D * sm

        sm = AD * sm

        sm = D * sm

        sm_ot = sm - ((dot(sm, on) / dot(on, on)) * on)

        sm_norm = sm_ot ./ norm(sm_ot);

        sm_vec[:, loop] = sm_norm[1:mx]

    end # for loop

    V = sm_vec[:, interval:interval:end]

    return V

end #end of function



## Input: hypergraph array, and a set of smoothed vectors
# Output: hyperedge scores
function HSC(ar, SV)
    score = zeros(eltype(SV), length(ar))
    @inbounds Threads.@threads for i in eachindex(ar)
        nodes = ar[i]
        for j in axes(SV, 2)
            mx, mn = -Inf, +Inf
            for node in nodes
                x = SV[node, j]
                mx = ifelse(x > mx, x, mx)
                mn = ifelse(x < mn, x, mn)
            end
            score[i] += (mx - mn)^2
        end
    end
    return score
end

## Input: hypergraph array
# Output: an array showing the hyperedges belong to each node
function HyperNodes(ar)

    H = INC(ar)

    NH1 = Any[]

    rr1 = H.rowval

    cc1 = H.colptr

    for i = 1:size(H, 2)

        st = cc1[i]

        ed = cc1[i+1] - 1

        push!(NH1, rr1[st:ed])

    end

    return NH1

end

## The output file shows the cluster that every nodes belong to it
function CLidx(idx_mat)
    V = 1:maximum(idx_mat[end])
    for ii = 1:length(idx_mat)
        idx1 = idx_mat[end - ii + 1]
        V = V[idx1]
    end
    open("Output","w")do io
        for ii = 1:length(V)
            println(io, V[ii])
        end
    end
end
