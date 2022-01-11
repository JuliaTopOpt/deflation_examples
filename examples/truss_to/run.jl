using GLMakie
using Makie
using TopOpt
using TopOpt.TrussTopOptProblems.TrussVisualization: visualize

Nonconvex.@load Ipopt

include("problem_defs.jl")
include("utils.jl")
using .DeflateTruss

function optimize_truss(opt_task; verbose=false, viz=true)
    force = [0,100.0]
    problem = CustomPointLoadCantileverTruss((40,10),Tuple(ones(2)),1.0,0.3,force)

    V = 0.1 # volume fraction
    xmin = 0.001 # minimum density
    rmin = 4.0 # density filter radius

    penalty = TopOpt.PowerPenalty(1.0)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    solver()

    comp = TopOpt.Compliance(problem, solver)
    obj = comp
    volfrac = TopOpt.Volume(problem, solver)
    constr = x -> volfrac(x) - V

    options = MMAOptions(; maxiter=3000, tol=Nonconvex.Tolerance(; kkt=0.001))
    convcriteria = Nonconvex.KKTCriteria()
    x0 = fill(V, length(solver.vars))
    nelem = length(x0)

    Nonconvex.NonconvexCore.show_residuals[] = verbose

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    alg = MMA87()

    TopOpt.setpenalty!(solver, penalty.p)
    # https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/mma/#NonconvexCore.Tolerance
    r1 = Nonconvex.optimize(m, alg, x0; options=options, convcriteria = convcriteria)
    println("Minimum: $(r1.minimum)")
    println("$(r1.convstate)")

    if viz
        fig = visualize(problem, topology=r1.minimizer)
        display(fig)
        wait_for_key("Press key to continue")
    end

    if opt_task == "compliance_deflation"
        xstar = r1.minimizer
        shift = 1.0
        power = 2.0
        function deflation_constr(X)
            # L-power distance
            return 1.0/norm(X[1:end-1]-xstar, power)^power + shift - X[end]
        end

        # * remake problem because dim changes by 1
        obj = x -> comp(x[1:end-1])
        constr = x -> volfrac(x[1:end-1]) - V

        m = Model(obj)
        addvar!(m, zeros(nelem), ones(nelem))
        # deflation slack variable y
        addvar!(m, [-1e3], [1e6])
        add_ineq_constraint!(m, constr)
        add_ineq_constraint!(m, deflation_constr)

        df_alg = IpoptAlg()
        df_options = IpoptOptions(tol = 1e-4)
        df_convcriteria = Nonconvex.KKTCriteria()
        # df_alg = alg
        # df_options = options
        # df_convcriteria = convcriteria

        r2 = optimize(m, df_alg, vcat(x0, 1.0), options = df_options, convcriteria = df_convcriteria)
        # println("$(r2.convstate)")
        println("Minimum: $(r2.minimum)")
    end

    if viz
        fig = visualize(problem, topology=r2.minimizer[1:end-1])
        display(fig)
        wait_for_key("Press key to continue")
    end
end

#########################################

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--opt", "-o"
            help = "type of optimization"
            arg_type = String
            default = "compliance_only"
        "--viz", "-v"
            help = "visualize results."
            action = :store_true
        "--verbose"
            help = "print extra info."
            action = :store_true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    println(parsed_args)
    println("="^10)

    optimize_truss(parsed_args["opt"], verbose=parsed_args["verbose"], viz=parsed_args["viz"])
end

main()