using DrWatson
quickactivate(@__DIR__)

using DataFrames, Statistics
using StatsPlots

p_range = 1.0:1:6.
r_range = 1.0:5:32

function expand_run_iters(df, colname)
    new_df = DataFrame()
    for row in eachrow(df)
        scores = row[colname]
        p, r = row[:power], row[:radius]
        for s in scores
            new_df = vcat(new_df, DataFrame("power"=>p, "radius"=>r, String(colname)=>s))
        end
    end
    new_df
end

df = DataFrame()
for (p, r) in Iterators.product(p_range, r_range)
    df = vcat(df, DataFrame("power"=>p, "radius"=>r, "score"=>[rand(20)]))
end
new_df = expand_run_iters(df, :score)

colorscheme = palette(:tab10, length(r_range))
fig = plot()
for (i, r) in enumerate(r_range)
    color = colorscheme[i]
    @df subset(new_df, :radius => x -> x .≈ r) boxplot!(:power, :score, label="r=$(r)", fillalpha=0.5, colour=color)
    # @df subset(new_df, :radius => x -> x .≈ r) violin!(:power, :score,  side=:left, linewidth=0, 
        # label="r=$(r)", fillalpha=0.2)
end
gui()