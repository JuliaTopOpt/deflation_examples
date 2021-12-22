mutable struct TrussSumFL{T} <: AbstractFunction{T}
    problem::TrussProblem{T}
    fevals::Int
    maxfevals::Int
end
# varying_node_ids::AbstractArray

Base.show(::IO, ::MIME{Symbol("text/plain")}, ::TrussSumFL) = println("TopOpt truss ∑ |F|L function")

"""
    TrussSumFL()

Construct the TrussSumFL function struct.
"""
function TrussSumFL(problem::TrussProblem; maxfevals = 10^8)
    return TrussSumFL(problem, 0, maxfevals)
end

"""
# Arguments
`x` = design variables

\sum \sigma * A * L ≈ required structural volume

# Returns
displacement vector `σ`, compressive stress < 0, tensile stress > 0
"""
function (ts::TrussSumFL)(x) where {T}
    # * update nodal positions
    @unpack problem = ts
    nodes = problem.truss_grid.grid.nodes
    # cnt = 1
    # for (node_id, changing_dofs) in varying_node_ids
    #     tmp_coord = collect(nodes[node_id].x)
    #     for c_dof in changing_dofs
    #         tmp_coord[c_dof] = x[cnt]
    #         cnt += 1
    #     end
    #     nodes[node_id] = Ferrite.Node((tmp_coord...,))
    # end
    # ! Ad-hoc changing node for now
    nodes[4] = Ferrite.Node((x...,))
    # ! Ad-hoc symmetry for now
    axis_node = collect(nodes[2].x)
    new_node = [axis_node[1] + axis_node[1]-nodes[4].x[1], nodes[4].x[2]]
    nodes[5] = Ferrite.Node((new_node...,))

    # * compute axial forces
    solver = FEASolver(Direct, problem)
    solver()
    solver.u
    stress_fn = TrussStress(solver)
    σ = stress_fn(ones(7))
    ts.fevals += 1
    # * compute ∑ abs(F)*L
    As = getA(problem)
    sumfl = 0.0
    dh = solver.problem.ch.dh
    for (e, cell) in enumerate(CellIterator(dh))
        u, v = cell.coords[1], cell.coords[2]
        L = LinearAlgebra.norm(u-v)
        sumfl += abs(σ[e]) * As[e] * L
    end
    return sumfl
end

function ChainRulesCore.rrule(ts::TrussSumFL, x::AbstractVector)
    val = ts(x)
    # grad = ForwardDiff.gradient(f, x)
    grad = FDM.grad(central_fdm(5, 1), ts, x)[1]
    val, Δ -> (NoTangent(), Δ * grad)
end