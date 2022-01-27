using JLD
##############################

wait_for_key(prompt) = (print(stdout, prompt); read(stdin, 1); nothing)

function replot_problem_results(problem, problem_result_dir)
    println("Loading data from $problem_result_dir")
    result_data = load(joinpath(problem_result_dir, "data.jld"))["result_data"]
    objs = Float64[]

    r1_x = result_data["init_solve"]["minimizer"]
    fig1 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=r1_x, undeformed_mesh_color=(:black, 1.0), 
        display_supports=false)
    hidedecorations!(fig1.current_axis.x)
    save(joinpath(problem_result_dir, "r1_replot.pdf"), fig1, pt_per_unit = 1)
    push!(objs, result_data["init_solve"]["minimum"] )

    for k in keys(result_data)
        if !occursin("deflate", k)
            continue
        end
        df_result_minimizer = result_data[k]["minimizer"]
        df_result_minimum = result_data[k]["minimum"]
        push!(objs, df_result_minimum)

        fig2 = TopOpt.TopOptProblems.Visualization.visualize(problem, topology=df_result_minimizer[1:end-1], 
            undeformed_mesh_color=(:black, 1.0), display_supports=false)
        hidedecorations!(fig2.current_axis.x)
        save(joinpath(problem_result_dir, "$(k)_replot.pdf"), fig2, pt_per_unit = 1)
    end

    objs ./= minimum(objs)
    obj_fig = Figure(resolution=(800,800), font="Arial")
    ax = Axis(obj_fig[1,1])
    # ax.xlabel = "deflation iterations"
    # ax.ylabel = "objectives"
    lines!(ax, 0:1:length(objs)-1, objs)
    save(joinpath(problem_result_dir, "_obj_history_replot.pdf"), obj_fig)
end