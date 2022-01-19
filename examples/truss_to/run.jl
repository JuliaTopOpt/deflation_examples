# using GLMakie
using CairoMakie
CairoMakie.activate!()
using Makie
using JLD

import Optim
using TopOpt
using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
Nonconvex.@load Ipopt
Nonconvex.@load NLopt
Nonconvex.@load Juniper

include("problem_defs.jl")
include("../utils.jl")
using .DeflateTruss

RESULT_DIR = joinpath(@__DIR__, "results");

function optimize_truss(problem, opt_task; verbose=false, write=false, optimizer="mma", distance="l2",
    deflation_iters=5)
    problem_config = "$(problem)_$(opt_task)_$(optimizer)_$distance"
    result_data = Dict()
    objs = Float64[]

    size_ratio = 1

    problem_result_dir = joinpath(RESULT_DIR, problem_config)
    if !ispath(problem_result_dir)
        mkdir(problem_result_dir)
    end

    if problem == "dense_graph"
        force = [0, 100.0]
        problem = CustomPointLoadCantileverTruss((40*size_ratio,10*size_ratio),Tuple(ones(2)),1.0,0.3,force)
    elseif problem == "tim"
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
            joinpath(@__DIR__, "tim_2d.json")
        );
        loads = load_cases["0"]
        problem = TrussProblem(
            Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
        );
    else
        error("Unsupported problem $problem")
    end

    V = 0.3 # volume fraction
    xmin = 0.0001 # minimum density
    p = 3.0

    penalty = TopOpt.PowerPenalty(p)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    solver()
    # TopOpt.setpenalty!(solver, penalty.p)

    comp = TopOpt.Compliance(problem, solver)

    if occursin("vol_constrained", opt_task) && occursin("min_compliance", opt_task)
        obj = comp
        volfrac = TopOpt.Volume(problem, solver)
        constr = x -> volfrac(x) - V
        # constr = x -> sum(x) / length(x) - V

        if occursin("buckling_constrained", opt_task)
            
        end
    else
        error("Undefined task $(opt_task)")
    end

    x0 = fill(V, length(solver.vars))
    nelem = length(x0)
    println("#elements : $nelem")

    Nonconvex.NonconvexCore.show_residuals[] = verbose

    m = Model(obj)
    if optimizer == "juniper"
        addvar!(m, zeros(nelem), ones(nelem); integer=trues(nelem))
    else
        addvar!(m, zeros(nelem), ones(nelem))
    end
    add_ineq_constraint!(m, constr)

    if optimizer == "mma"
        alg = MMA02()
        options = MMAOptions(maxiter=100, 
            tol=Nonconvex.Tolerance(kkt=0.001),
            # dual_options=Optim.Options(iterations = 100)
            )
    elseif optimizer == "juniper"
        # https://lanl-ansi.github.io/Juniper.jl/stable/options/#Parallel
        alg = JuniperIpoptAlg()
        options = JuniperIpoptOptions(subsolver_options = IpoptOptions(max_iter=300))
    elseif optimizer == "nlopt"
        alg = NLoptAlg(:LD_MMA)
        options = NLoptOptions()
    elseif optimizer == "ipopt"
        alg = IpoptAlg()
        options = IpoptOptions(tol = 1e-4, max_iter=1000)
    else
        error("Undefined optimizer $optimizer")
    end
    convcriteria = Nonconvex.KKTCriteria()

    println("Initial solve:")
    # https://julianonconvex.github.io/Nonconvex.jl/stable/algorithms/mma/#NonconvexCore.Tolerance
    r1 = Nonconvex.optimize(m, alg, x0; options=options, convcriteria = convcriteria)
    # println("$(r1.convstate)")
    println("Minimum: $(r1.minimum); Constraint: $(maximum(constr(r1.minimizer)))")

    fig1 = visualize(problem, topology=r1.minimizer)
    hidedecorations!(fig1.current_axis.x)
    # var_fig = lines(r1.minimizer)
    push!(objs, r1.minimum)
    if write
        result_data["init_solve"] = Dict("minimizer" => r1.minimizer, "minimum" => r1.minimum)
         #, "convstate" => r1.convstate)
        save(joinpath(problem_result_dir, "r1.png"), fig1)
        # https://makie.juliaplots.org/stable/documentation/figure_size/#vector_graphics
        save(joinpath(problem_result_dir, "r1.pdf"), fig1, pt_per_unit = 1)
        # save(joinpath(problem_result_dir, "x1.png"), var_fig)
    else
        display(fig1)
        wait_for_key("Enter to continue...")
        # display(var_fig)
        # wait_for_key("Check variable distribution...")
    end

    if endswith(opt_task, "deflation")
        shift = 1.0
        power = 4.0
        radius = 20.0

        if distance == "l2"
            dist_fn = (x,y) -> norm(x-y, 2)
        elseif distance == "kl"
            dist_fn = (x,y) -> multi_bernoulli_kl_divergence(x,y)
        end

        solutions = [r1.minimizer]
        for iter in 1:deflation_iters
            println("-"^10)
            println("Iter - $iter")
            function deflation_constr(X::Vector)
                d = zero(eltype(X))
                for sol in solutions
                    d += max(dist_fn(X[1:end-1], sol) - radius, 0)^(-power)
                    # d += dist_fn(X[1:end-1], sol)^(-power) + shift
                end
                println("d = $d, y = $(X[end])")
                return d - X[end]
            end

            # * remake problem because dim changes by 1
            df_obj = x -> obj(x[1:end-1])
            df_constr = x -> constr(x[1:end-1])

            m = Model(df_obj)
            if optimizer == "juniper"
                addvar!(m, zeros(nelem), ones(nelem); integer=trues(nelem))
            else
                addvar!(m, zeros(nelem), ones(nelem))
            end
            # * deflation slack variable y
            addvar!(m, [0], [1e2])
            add_ineq_constraint!(m, df_constr)
            add_ineq_constraint!(m, deflation_constr)

            df_alg = alg
            df_options = 
                # MMAOptions(
                # # s_incr=1.0, s_decr=1.0, s_init = 0.1,
                # maxiter=100, tol=Nonconvex.Tolerance(kkt=0.001), 
                # dual_options=Optim.Options(iterations=100))
                options
            df_convcriteria = convcriteria

            df_result = optimize(m, df_alg, vcat(x0, 0.0), options = df_options, convcriteria = df_convcriteria)
            # println("$(r2.convstate)")
            println("Minimum: $(df_result.minimum); Constraint: $(maximum(df_constr(df_result.minimizer))); DF Constraint: $(maximum(deflation_constr(df_result.minimizer)))")
            final_dist = dist_fn(r1.minimizer, df_result.minimizer[1:end-1])
            println("Distance: $(final_dist)")

            push!(objs, df_result.minimum)
            fig2 = visualize(problem, topology=df_result.minimizer[1:end-1])
            hidedecorations!(fig2.current_axis.x)
            # var_fig = lines(df_result.minimizer[1:end-1])
            if write
                result_data["deflate_$(iter)"] = Dict("minimizer" => df_result.minimizer, "minimum" => df_result.minimum) #, "convstate" => r1.convstate)
                save(joinpath(problem_result_dir, "deflate_r$(iter).png"), fig2)
                save(joinpath(problem_result_dir, "deflate_r$(iter).pdf"), fig2, pt_per_unit = 1)
                # save(joinpath(problem_result_dir, "_deflate_x$(iter).png"), var_fig)
            else
                display(fig2)
                wait_for_key("Enter to continue...")
                # display(var_fig)
                # wait_for_key("Check variable distribution...")
            end

            push!(solutions, df_result.minimizer[1:end-1])
        end

        if write
            obj_fig = lines(objs)
            save(joinpath(problem_result_dir, "_obj_history.png"), obj_fig)
        end
    end

    if write
        save(joinpath(problem_result_dir, "data.jld"), "result_data", result_data)
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
            default = "dense_graph" # tim
        "--task"
            help = "problem formulation"
            arg_type = String
            default = "min_compliance_vol_constrained_deflation" # _deflation, _buckling_constrained
        "--optimizer"
            help = "optimizer choice"
            arg_type = String
            default = "mma" # nlopt, juniper, ipopt
        "--distance"
            help = "distance measure choice"
            arg_type = String
            default = "l2" # kl, w2
        "--deflate_iters"
            help = "print extra info."
            arg_type = Int
            default = 5
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

    # TODO try https://juliadynamics.github.io/DrWatson.jl/dev/real_world/
    optimize_truss(parsed_args["problem"], parsed_args["task"], verbose=parsed_args["verbose"],
        write=parsed_args["write"], optimizer=parsed_args["optimizer"], distance=parsed_args["distance"],
        deflation_iters=parsed_args["deflate_iters"])
end

main()