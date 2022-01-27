using GLMakie
# using CairoMakie
GLMakie.activate!()
using Makie
using JLD
using Formatting

import Optim
using TopOpt
using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
Nonconvex.@load Ipopt
Nonconvex.@load NLopt

include("problem_defs.jl")
include("../utils.jl")
using .DeflateTruss

RESULT_DIR = joinpath(@__DIR__, "results");

function optimize_truss(problem_name, opt_task; verbose=false, write=false, optimizer="mma", distance="l2",
    deflation_iters=5)
    result_data = Dict()
    objs = Float64[]

    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    size_ratio = 1
    if problem_name == "dense_graph"
        f = [0.0, 100.0]
        problem = CustomPointLoadCantileverTruss((40*size_ratio,10*size_ratio),Tuple(ones(2)),E,v,f)
    elseif problem == "tim"
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
            joinpath(@__DIR__, "tim_2d.json")
        );
        loads = load_cases["0"]
        f = "load case 0"
        problem = TrussProblem(
            Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
        );
    else
        error("Unsupported problem $problem_name")
    end

    problem_config = "$(problem_name)_$(opt_task)_$(optimizer)_$distance"
    problem_result_dir = joinpath(RESULT_DIR, problem_config)
    if !ispath(problem_result_dir)
        mkdir(problem_result_dir)
    end

    V = 0.3 # volume fraction
    xmin = 0.0001 # minimum density
    penalty_val = 3.0

    penalty = TopOpt.PowerPenalty(penalty_val)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    solver()
    result_data["config"] = Dict("E" => E, "ν" => v, "f" => f, "penalty" => penalty_val, "xmin" => xmin, "vol_fraction" => V)

    # * comliance minimization objective
    if occursin("vol_constrained", opt_task) && occursin("min_compliance", opt_task)
        comp = TopOpt.Compliance(problem, solver)
        obj = comp
        volfrac = TopOpt.Volume(problem, solver)
        constr = x -> volfrac(x) - V
        # or equivalently:
        # constr = x -> sum(x) / length(x) - V
    else
        error("Undefined task $(opt_task)")
    end

    x0 = fill(V, length(solver.vars))
    nelem = length(x0)
    println("#elements : $nelem")

    Nonconvex.NonconvexCore.show_residuals[] = verbose

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    if optimizer == "mma"
        alg = MMA02()
        options = MMAOptions(maxiter=100, 
            tol=Nonconvex.Tolerance(kkt=0.001),
            # dual_options=Optim.Options(iterations = 100)
            )
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
    st_time = time()
    r1 = Nonconvex.optimize(m, alg, x0; options=options, convcriteria = convcriteria)
    runtime = time() - st_time
    printfmt("Minimum: {:.3f}; Constraint: {:.3f}; runtime {:.3f}\n", r1.minimum, maximum(constr(r1.minimizer)), runtime)

    fig1 = visualize(problem, topology=r1.minimizer)
    hidedecorations!(fig1.current_axis.x)
    push!(objs, r1.minimum)

    if write
        result_data["init_solve"] = Dict("minimizer" => r1.minimizer, "minimum" => r1.minimum, "runtime" => runtime)
        save(joinpath(problem_result_dir, "r1.png"), fig1)
        # save(joinpath(problem_result_dir, "r1.pdf"), fig1, pt_per_unit = 1)
    else
        display(fig1)
        wait_for_key("Enter to continue...")
    end

    if endswith(opt_task, "deflation")
        power = 4.0
        radius = 20.0
        y_upperbound = 1e2
        result_data["deflation_config"] = Dict("power" => power, "radius" => radius, "dist" => distance,
            "y_upperbound" => y_upperbound)

        if distance == "l2"
            dist_fn = (x,y) -> norm(x-y, 2)
        else
            error("Unimplemented distance $distance")
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
            addvar!(m, zeros(nelem), ones(nelem))
            # * deflation slack variable y
            addvar!(m, [0], [y_upperbound])
            add_ineq_constraint!(m, df_constr)
            add_ineq_constraint!(m, deflation_constr)

            df_alg = alg
            df_options = options
            df_convcriteria = convcriteria

            st_time = time()
            df_result = optimize(m, df_alg, vcat(x0, 0.0), options = df_options, convcriteria = df_convcriteria)
            runtime = time() - st_time

            printfmt("Minimum: {:.3f}; Constraint: {:.3f}; DF Constraint: {:.3f}; runtime {:.3f}\n", 
                df_result.minimum, maximum(df_constr(df_result.minimizer)), 
                maximum(deflation_constr(df_result.minimizer)), runtime)
            final_dist = dist_fn(r1.minimizer, df_result.minimizer[1:end-1])
            println("Distance: $(final_dist)")
            push!(objs, df_result.minimum)

            fig2 = visualize(problem, topology=df_result.minimizer[1:end-1])
            hidedecorations!(fig2.current_axis.x)

            if write
                result_data["deflate_$(iter)"] = Dict("minimizer" => df_result.minimizer, "minimum" => df_result.minimum, "runtime" => runtime, "distance_to_xstar" => final_dist)
                save(joinpath(problem_result_dir, "deflate_r$(iter).png"), fig2)
                # save(joinpath(problem_result_dir, "deflate_r$(iter).pdf"), fig2, pt_per_unit = 1)
            else
                display(fig2)
                wait_for_key("Enter to continue...")
            end
            push!(solutions, df_result.minimizer[1:end-1])
        end

        if write
            obj_fig = Figure(resolution=(800,800), font="Arial")
            ax = Axis(obj_fig[1,1])
            ax.xlabel = "deflation iterations"
            ax.ylabel = "objectives"
            lines!(ax, 0:1:length(objs)-1, objs)
            # save(joinpath(problem_result_dir, "_obj_history.pdf"), obj_fig)
            save(joinpath(problem_result_dir, "_obj_history.png"), obj_fig)
        end
    end # end deflation loop

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
            default = "nlopt" # mma, nlopt, ipopt
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
        "--replot"
            help = "parse and replot saved results."
            action = :store_true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    println(parsed_args)
    println("="^10)

    optimize_truss(parsed_args["problem"], parsed_args["task"], verbose=parsed_args["verbose"],
        write=parsed_args["write"], optimizer=parsed_args["optimizer"], distance=parsed_args["distance"],
        deflation_iters=parsed_args["deflate_iters"])
end

main()