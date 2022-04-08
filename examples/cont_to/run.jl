using GLMakie
# using CairoMakie
GLMakie.activate!()
using Makie
using JLD
using Formatting

using TopOpt
using TopOpt.TopOptProblems.Visualization: visualize
using LinearAlgebra, StatsFuns

include("../utils.jl")

using Nonconvex
Nonconvex.@load Ipopt
Nonconvex.@load NLopt

RESULT_DIR = joinpath(@__DIR__, "results");

function optimize_domain(problem_name, opt_task; verbose=false, write=false, optimizer="nlopt", distance="l2",
    deflation_iters=5, replot=false)
    result_data = Dict()
    objs = Float64[]

    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    f = 1.0 # downward force
    rmin = 4.0
    size_ratio = 2
    problems = Dict(
        "cantilever_beam" => PointLoadCantilever(Val{:Linear}, (80*size_ratio, 20*size_ratio), (1.0, 1.0), E, v, f),
        "half_mbb_beam" => HalfMBB(Val{:Linear}, (60*size_ratio, 20*size_ratio), (1.0, 1.0), E, v, f),
    )
    problem = problems[problem_name]
    problem_config = "$(problem_name)_$(opt_task)_$(optimizer)_$distance"

    problem_result_dir = joinpath(RESULT_DIR, problem_config)
    if !ispath(problem_result_dir)
        mkdir(problem_result_dir)
    end

    if replot
        replot_problem_results(problem, problem_result_dir)
        println("Replotting done.")
        return
    end

    V = 0.5 # volume fraction
    xmin = 0.0001 # minimum density
    penalty_val = 4.0
    penalty = TopOpt.PowerPenalty(penalty_val)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    TopOpt.setpenalty!(solver, penalty.p)
    filter = TopOpt.DensityFilter(solver; rmin=rmin)
    result_data["config"] = Dict("E" => E, "ν" => v, "f" => f, "rmin" => rmin,
        "penalty" => penalty_val, "xmin" => xmin, "vol_fraction" => V)

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
    elseif occursin("min_vol", opt_task) && occursin("compliance_constrained", opt_task)
        # minimize volume
        obj = x -> sum(filter(x)) / length(x) - V
        comp = TopOpt.Compliance(problem, solver)
        compliance_threshold = 1500 # maximum compliance
        # <= 0
        constr = x -> comp(filter(x)) - compliance_threshold 
    else
        error("Undefined task $(opt_task)")
    end

    Nonconvex.NonconvexCore.show_residuals[] = verbose

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    if optimizer == "mma"
        alg = MMA87()
        options = MMAOptions(; maxiter=100, tol=Nonconvex.Tolerance(; kkt=0.001))
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

    fig1 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=r1.minimizer, undeformed_mesh_color=(:black, 1.0))
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

    if endswith(opt_task, "deflation") || endswith(opt_task, "random_restart")
        power = 4.0
        radius = 30.0
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

            st_time = time()
            if endswith(opt_task, "deflation")
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
                addvar!(m, zeros(nelem), ones(nelem); integer=trues(nelem))
                # * deflation slack variable y
                addvar!(m, [0], [y_upperbound])
                add_ineq_constraint!(m, df_constr)
                add_ineq_constraint!(m, deflation_constr)

                df_alg = alg
                df_options = options
                df_convcriteria = convcriteria

                # * start from the same initial solution!
                df_result = optimize(m, df_alg, vcat(x0, 1.0), options = df_options, convcriteria = df_convcriteria)
                new_topology = df_result.minimizer[1:end-1]
                new_minimum = df_result.minimum
                final_dist = dist_fn(r1.minimizer, df_result.minimizer[1:end-1])

                runtime = time() - st_time
                printfmt("Minimum: {:.3f}; Constraint: {:.3f}; DF Constraint: {:.3f}; runtime {:.3f}\n", 
                    df_result.minimum, maximum(df_constr(df_result.minimizer)), 
                    maximum(deflation_constr(df_result.minimizer)), runtime)
                println("Distance: $(final_dist)")
                data_to_store = Dict("minimizer" => df_result.minimizer, "minimum" => df_result.minimum, "runtime" => runtime, 
                    "distance_to_xstar" => final_dist)
            elseif endswith(opt_task, "random_restart")
                rs_x0 = rand(eltype(V), length(solver.vars))
                rs_result = Nonconvex.optimize(m, alg, rs_x0; options=options, convcriteria = convcriteria)
                runtime = time() - st_time
                printfmt("Minimum: {:.3f}; Constraint: {:.3f}; runtime {:.3f}\n", rs_result.minimum, 
                    maximum(constr(rs_result.minimizer)), runtime)

                new_topology = rs_result.minimizer
                new_minimum = rs_result.minimum
                final_dist = dist_fn(r1.minimizer, rs_result.minimizer)

                data_to_store = Dict("minimizer" => rs_result.minimizer, "minimum" => rs_result.minimum, "runtime" => runtime, 
                    "distance_to_xstar" => final_dist)
            else
                error("unknown task $(opt_task)")
            end

            push!(objs, new_minimum)
            push!(solutions, new_topology)

            fig2 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=new_topology, 
                undeformed_mesh_color=(:black, 1.0))
            hidedecorations!(fig2.current_axis.x)

            if write
                result_data["iter_$(iter)"] = data_to_store
                save(joinpath(problem_result_dir, "iter_r$(iter).png"), fig2)
                # save(joinpath(problem_result_dir, "deflate_r$(iter).pdf"), fig2, pt_per_unit = 1)
            else
                display(fig2)
                wait_for_key("Enter to continue...")
            end
        end # end deflation/random_restart loop

        if write
            obj_fig = Figure(resolution=(800,800), font="Arial")
            ax = Axis(obj_fig[1,1])
            ax.xlabel = "iterations"
            ax.ylabel = "objectives"
            lines!(ax, 0:1:length(objs)-1, objs/objs[1])
            # save(joinpath(problem_result_dir, "_obj_history.pdf"), obj_fig)
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
            default = "half_mbb_beam"
        "--task"
            help = "problem formulation"
            arg_type = String
            default = "min_compliance_vol_constrained_deflation" # "min_vol_stress_constrained_deflation"
        "--optimizer"
            help = "optimizer choice"
            arg_type = String
            default = "nlopt" # mma
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

    optimize_domain(parsed_args["problem"], parsed_args["task"], verbose=parsed_args["verbose"],
        write=parsed_args["write"], optimizer=parsed_args["optimizer"], distance=parsed_args["distance"],
        deflation_iters=parsed_args["deflate_iters"], replot=parsed_args["replot"])
end

main()