using DrWatson
quickactivate(@__DIR__)

using DataFrames, Statistics
using StatsPlots

p_range = 1.0:1:6.
r_range = 1.0:5:32

function expand_run_iters(df)
    new_df = DataFrame()
    for row in eachrow(df)
        # the first one is the initial solve
        scores = row[:relative_objective_history][2:end]
        distances = row[:distance_to_initial_solution][2:end]
        runtimes = row[:relative_runtimes][2:end]
        p, r = row[:power], row[:radius]
        for (objective, dist, runtime) in zip(scores, distances, runtimes)
            run_data = @strdict objective dist runtime
            new_df = vcat(new_df, DataFrame("power"=>p, "radius"=>r, run_data...))
        end
    end
    new_df
end

df = collect_results(datadir("cont_to", "hyperparam_tests"))
new_df = expand_run_iters(df)

# https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/
colorscheme = palette(:terrain, length(r_range))
for (i, r) in enumerate(r_range)
    plots = []
    color = colorscheme[i]
    sub_df = subset(new_df, :radius => x -> x .≈ r) #, :objective => x->x .< 10)
    selected_subdf = sub_df

    subfig = plot()
    @df selected_subdf boxplot!(:power, :objective, fillalpha=0.5, colour=color)
    @df selected_subdf violin!(:power, :objective, side=:left, linewidth=0, fillalpha=0.2, legend=false)
    ylims!(subfig, (1.0, 1.2))
    xlabel!(subfig, "power")
    ylabel!(subfig, "relative objectives")
    push!(plots, subfig)

    subfig = plot()
    @df selected_subdf boxplot!(:power, :dist, fillalpha=0.5, colour=color, legend=false)
    @df selected_subdf violin!(:power, :dist,  side=:left, linewidth=0, fillalpha=0.2, legend=false)
    ylims!(subfig, (0.0, 55.))
    xlabel!(subfig, "power")
    ylabel!(subfig, "distance to initial solution")
    push!(plots, subfig)

    subfig = plot()
    @df selected_subdf boxplot!(:power, :runtime, fillalpha=0.5, colour=color, legend=false)
    @df selected_subdf violin!(:power, :runtime,  side=:left, linewidth=0, fillalpha=0.2, legend=false)
    ylims!(subfig, (0.0, 5.5))
    xlabel!(subfig, "power")
    ylabel!(subfig, "relative runtime")
    push!(plots, subfig)

    fig = plot(plots..., layout=(1, length(plots)), plot_title="radius=$(r)")
    # fig[:plot_title] = ""
    wsave(plotsdir("cont_to", "hyperparam_tests", "objective_r=$r.png"), fig)
end
