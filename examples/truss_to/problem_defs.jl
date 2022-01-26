module DeflateTruss

using Ferrite
using StaticArrays
using NearestNeighbors
using TopOpt.TopOptProblems: StiffnessTopOptProblem, Metadata, RectilinearGrid, 
    left, right, middley, middlez, find_black_and_white, find_varind
using TopOpt.TrussTopOptProblems
using TopOpt.TrussTopOptProblems: TrussProblem, get_fixities_node_set_name

export CustomPointLoadCantileverTruss

######################

function CustomPointLoadCantileverTruss(nels::NTuple{dim,Int}, sizes::NTuple{dim}, E = 1.0, ν = 0.3, force = ones(dim); k_connect=1) where {dim}
    iseven(nels[2]) && (length(nels) < 3 || iseven(nels[3])) || throw("Grid does not have an even number of elements along the y and/or z axes.")
    _T = promote_type(eltype(sizes), typeof(E), typeof(ν), eltype(force))
    if _T <: Integer
        T = Float64
    else
        T = _T
    end
    @assert length(force) == dim

    # only for the convience of getting all the node points
    rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    node_mat = hcat(map(x -> Vector(x.x), rect_grid.grid.nodes)...)
    kdtree = KDTree(node_mat)
    if dim == 2
        # 4+1*4 -> 4+3*4 -> 4+5*4
        k_ = 4*k_connect + 4*sum(1:2:(2*k_connect-1))
    else
        k_ = 8*k_connect + 6*sum(1:9:(9*k_connect-1))
    end
    idxs, _ = knn(kdtree, node_mat, k_+1, true)
    connect_mat = zeros(Int, 2, k_*length(idxs))
    for (i, v) in enumerate(idxs)
        connect_mat[1, (i-1)*k_+1:i*k_] = ones(Int, k_)*i
        connect_mat[2, (i-1)*k_+1:i*k_] = v[2:end] # skip the point itself
    end
    truss_grid = TrussGrid(node_mat, connect_mat)

    # reference domain dimension for a line element
    ξdim = 1
    ncells = getncells(truss_grid)
    mats = [TrussFEAMaterial{T}(E, ν) for i=1:ncells]

    # * support nodeset
    for i in 1:dim
        addnodeset!(truss_grid.grid, get_fixities_node_set_name(i), x -> left(rect_grid, x));
    end

    # * load nodeset
    if dim == 2
        addnodeset!(truss_grid.grid, "force", x -> right(rect_grid, x) && middley(rect_grid, x));
    else
        addnodeset!(truss_grid.grid, "force", x -> right(rect_grid, x) && middley(rect_grid, x)
            && middlez(rect_grid, x));
    end

    # * Create displacement field u
    geom_order = 1
    dh = DofHandler(truss_grid.grid)
    ip = Lagrange{ξdim, RefCube, geom_order}()
    push!(dh, :u, dim, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    for i in 1:dim
        dbc = Dirichlet(
            :u,
            getnodeset(truss_grid.grid, get_fixities_node_set_name(i)),
            (x, t) -> zeros(T, 1),
            [i],
        )
        add!(ch, dbc)
    end
    close!(ch)
    # update the DBC to current time
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    loadset = getnodeset(truss_grid.grid, "force")
    ploads = Dict{Int, SVector{dim, T}}()
    for node_id in loadset
        ploads[node_id] = SVector{dim,T}(dim == 2 ? force : force)
    end

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    return TrussProblem(truss_grid, mats, ch, ploads, black, white, varind, metadata)
end

end # end module