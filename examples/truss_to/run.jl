using GLMakie
using Makie
using JLD

using TopOpt
using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
Nonconvex.@load Ipopt

include("problem_defs.jl")
include("utils.jl")
using .DeflateTruss

RESULT_DIR = joinpath(@__DIR__, "results");

function optimize_truss(opt_task; verbose=false, viz=true, write=false)
    result_data = Dict()
    force = [0,100.0]
    problem = CustomPointLoadCantileverTruss((40,10),Tuple(ones(2)),1.0,0.3,force)

    V = 0.1 # volume fraction
    xmin = 0.001 # minimum density

    penalty = TopOpt.PowerPenalty(1.0)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    solver()

    comp = TopOpt.Compliance(problem, solver)
    volfrac = TopOpt.Volume(problem, solver)

    # TODO other formulations
    obj = comp
    constr = x -> volfrac(x) - V

    options = MMAOptions(; maxiter=3, tol=Nonconvex.Tolerance(; kkt=0.001))
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

    fig1 = visualize(problem, topology=r1.minimizer)
    if write
        result_data["r1"] = Dict("minimizer" => r1.minimizer, "minimum" => r1.minimum, "convstate" => r1.convstate)
        save(joinpath(RESULT_DIR, "$(opt_task)_r1.png"), fig1)
    end
    if viz
        display(fig1)
        wait_for_key("Press key to continue")
    end

    if endswith(opt_task, "deflation")
        xstar = r1.minimizer
        shift = 1.0
        power = 4.0
        function deflation_constr(X)
            # L-2 distance
            return 1.0/norm(X[1:end-1]-xstar, 2)^power + shift - X[end]
        end

        # * remake problem because dim changes by 1
        df_obj = x -> obj(x[1:end-1])
        df_constr = x -> constr(x[1:end-1])

        m = Model(df_obj)
        addvar!(m, zeros(nelem), ones(nelem))
        # deflation slack variable y
        addvar!(m, [-1e3], [1e6])
        add_ineq_constraint!(m, df_constr)
        add_ineq_constraint!(m, deflation_constr)

        # default maxiter = 3000
        df_alg = IpoptAlg()
        df_options = IpoptOptions(tol = 1e-4)
        df_convcriteria = Nonconvex.KKTCriteria()
        # df_alg = alg
        # df_options = options
        # df_convcriteria = convcriteria

        r2 = optimize(m, df_alg, vcat(x0, 1.0), options = df_options, convcriteria = df_convcriteria)
        # println("$(r2.convstate)")
        println("Minimum: $(r2.minimum)")

        fig2 = visualize(problem, topology=r2.minimizer[1:end-1])
        if write
            result_data["r2"] = Dict("minimizer" => r2.minimizer, "minimum" => r2.minimum) #, "convstate" => r1.convstate)
            save(joinpath(RESULT_DIR, "$(opt_task)_r2.png"), fig2)
        end
        if viz
            display(fig2)
            wait_for_key("Press key to continue")
        end
    end

    if write
        save(joinpath(RESULT_DIR, "$(opt_task).jld"), "result_data", result_data)
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
        "--write"
            help = "export result."
            action = :store_true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    println(parsed_args)
    println("="^10)

    optimize_truss(parsed_args["opt"], verbose=parsed_args["verbose"], viz=parsed_args["viz"], 
        write=parsed_args["write"])
end

main()