using GLMakie
GLMakie.activate!()
# using CairoMakie
# CairoMakie.activate!()
using Makie
using JLD

using TopOpt
# using TopOpt.TopOptProblems.Visualization: visualize
using LinearAlgebra, StatsFuns
using StatsFuns: logsumexp

include("../utils.jl")

using Nonconvex
Nonconvex.@load Percival

RESULT_DIR = joinpath(@__DIR__, "results");

function optimize_domain(problem_name, opt_task; verbose=false, write=false, optimizer="percival", distance="")
    result_data = Dict()

    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    f = 1.0 # downward force
    rmin = 4.0
    size_ratio = 1
    problems = Dict(
        "cantilever_beam" => PointLoadCantilever(Val{:Linear}, (80*size_ratio, 20*size_ratio), (1.0, 1.0), E, v, f),
        "half_mbb_beam" => HalfMBB(Val{:Linear}, (60*size_ratio, 20*size_ratio), (1.0, 1.0), E, v, f),
        "l_beam" => LBeam(Val{:Linear}, Float64),
    )
    problem = problems[problem_name]
    problem_config = "$(problem_name)_$(optimizer)_$distance"

    V = 0.5 # volume fraction
    xmin = 0.0001 # minimum density
    p = 4.0
    penalty = TopOpt.PowerPenalty(p)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    TopOpt.setpenalty!(solver, penalty.p)
    filter = TopOpt.DensityFilter(solver; rmin=rmin)

    x0 = fill(V, length(solver.vars))
    nelem = length(x0)
    println("#elements : $nelem")

    if occursin("vol_constrained", opt_task) && occursin("min_compliance", opt_task)
        comp = TopOpt.Compliance(problem, solver)
        function obj(x)
            # minimize compliance
            return comp(filter(x))
        end
        function constr(x)
            # volume fraction constraint
            return sum(filter(x)) / length(x) - V
        end
    elseif occursin("min_vol", opt_task)
        # minimize volume
        obj = x -> sum(filter(x)) / length(x) - V

        if occursin("stress_constrained", opt_task)
            stress = TopOpt.MicroVonMisesStress(solver)
            constr = x -> begin
                s = stress(filter(x))
                thr = 10
                vcat((s .- thr) / 100, logsumexp(s) - log(length(s)) - thr)
            end
        elseif occursin("compliance_constrained", opt_task)
            comp = TopOpt.Compliance(problem, solver)
            compliance_threshold = 1500 # maximum compliance
            # <= 0
            constr = x -> comp(filter(x)) - compliance_threshold 
        else
            error("Undefined task $(opt_task)")
        end
    else
        error("Undefined task $(opt_task)")
    end

    Nonconvex.NonconvexCore.show_residuals[] = verbose

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    if optimizer == "percival"
        alg = AugLag()
        options = AugLagOptions()
    elseif optimizer == "mma"
        alg = MMA87()
        options = MMAOptions(; maxiter=100, tol=Nonconvex.Tolerance(; kkt=0.001))
    else
        error("Undefined optimizer $optimizer")
    end
    convcriteria = Nonconvex.KKTCriteria()

    r1 = Nonconvex.optimize(m, alg, x0; options=options, convcriteria = convcriteria)
    println("Minimum: $(r1.minimum); Constraint: $(maximum(constr(r1.minimizer)))")

    # fig1 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=filter(r1.minimizer))
    fig1 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=r1.minimizer)
    hidedecorations!(fig1.current_axis.x)
    var_fig = lines(r1.minimizer)
 
    if write
        result_data["r1"] = Dict("minimizer" => r1.minimizer, "minimum" => r1.minimum)
        save(joinpath(RESULT_DIR, "$(problem_config)_r1.png"), fig1)
        save(joinpath(RESULT_DIR, "$(problem_config)_x1.png"), var_fig)
        # save(joinpath(RESULT_DIR, "$(problem_config)_r1.pdf"), fig1, pt_per_unit = 1)
    else
        display(fig1)
        wait_for_key("Enter to continue...")
        display(var_fig)
        wait_for_key("Check variable distribution...")
    end

    if endswith(opt_task, "deflation")
        xstar = r1.minimizer
        shift = 1.0
        power = 4.0
        radius = 0.0
        function deflation_constr(X)
            # L-2 distance
            # return 1.0/norm(X[1:end-1]-xstar, 2)^power + shift - X[end]
            return max(norm(X[1:end-1]-xstar, 2)- radius, 0)^(-power) - X[end]
        end

        # * remake problem because dim changes by 1
        df_obj = x -> obj(x[1:end-1])
        df_constr = x -> constr(x[1:end-1])

        m = Model(df_obj)
        addvar!(m, zeros(nelem), ones(nelem); integer=trues(nelem))
        # * deflation slack variable y
        addvar!(m, [-1e2], [1e2])
        add_ineq_constraint!(m, df_constr)
        add_ineq_constraint!(m, deflation_constr)

        df_alg = alg
        df_options = options
        df_convcriteria = convcriteria

        r2 = optimize(m, df_alg, vcat(x0, 1.0), options = df_options, convcriteria = df_convcriteria)
        println("Minimum: $(r2.minimum); Constraint: $(maximum(df_constr(r2.minimizer)))")

        fig2 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=r2.minimizer[1:end-1])
        hidedecorations!(fig2.current_axis.x)
        var_fig = lines(r2.minimizer)

        if write
            result_data["r2"] = Dict("minimizer" => r2.minimizer, "minimum" => r2.minimum) #, "convstate" => r1.convstate)
            save(joinpath(RESULT_DIR, "$(problem_config)_r2.png"), fig2)
            save(joinpath(RESULT_DIR, "$(problem_config)_x2.png"), var_fig)
            # save(joinpath(RESULT_DIR, "$(problem_config)_r2.pdf"), fig2, pt_per_unit = 1)
        else
            display(fig2)
            wait_for_key("Enter to continue...")
            display(var_fig)
            wait_for_key("Check variable distribution...")
        end
    end
end

#########################################

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--problem"
            help = "problem to run"
            arg_type = String
            default = "cantilever_beam"
        "--task"
            help = "problem formulation"
            arg_type = String
            default = "min_vol_stress_constrained_deflation"
        "--optimizer"
            help = "optimizer choice"
            arg_type = String
            default = "percival" #
        "--distance"
            help = "distance measure choice"
            arg_type = String
            default = "l2" # kl, w2
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

    optimize_domain(parsed_args["problem"], parsed_args["task"], verbose=parsed_args["verbose"],
        write=parsed_args["write"], optimizer=parsed_args["optimizer"], distance=parsed_args["distance"])
end

main()