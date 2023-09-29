using Revise, StableRNGs, Pathfinder, Distributions, ForwardDiff, Plots, Optim, LineSearches, NonconvexIpopt, Random

function run_pathfinder(; nruns = 10, ndraws = 1000_000, ndraws_per_run = ndraws ÷ nruns, ndists = 2, rng = StableRNG(123), means = 100 * randn(rng, ndists), initx = [1000.0], power = 3.0, shift = 0.0, deflation_ub = 1000, deflation_radius = 0.0, plot_title = "", reduce = +)
    function mixturef(means, stds = ones(length(means)))
        return MixtureModel(Normal.(means, stds))
    end
    mixture = mixturef(means)
    logπ = x -> logpdf(mixture, x[1])
    ∇logπ = x -> ForwardDiff.gradient(logπ, x)
    plt = Plots.plot(logπ, minimum(means) - 5, maximum(means) + 5, label = "target pdf", legend = :outertopright, title = plot_title, linewidth = 3.0)
    @show means

    # x₀s = [30 * randn(rng, length(initx)) for _ in 1:nruns]
    x₀s = [initx for _ in 1:nruns]

    # Deflation
    optimizer1 = IpoptAlg()
    q1, ϕ1, component_ids1 = multipathfinder(
        logπ, ∇logπ, x₀s, ndraws; ndraws_elbo=100, ndraws_per_run=ndraws_per_run, optimizer=optimizer1,
        rng, power, deflation_ub, deflation_radius, reduce, shift,
    )

    logπ2 = x -> logpdf(q1, [x])
    Plots.plot!(plt, logπ2, minimum(means) - 5, maximum(means) + 5, label = "approx", linewidth = 3.0)

    return q1, ϕ1, component_ids1, plt
end

RESULT_DIR = joinpath(@__DIR__, "results");
for ndists in [2, 4, 6, 8]
    nruns = 300
    ndraws = 10000
    rng = StableRNG(1234)
    means = 30 * randn(rng, ndists)
    initx = [0.0]
    deflation_radius = 0.0
    reduce_op = +
    power = 4.0
    shift = 0.3

    q1, ϕ1, component_ids1, plt = run_pathfinder(; nruns, ndraws, ndists, rng, means, initx, deflation_radius, reduce = reduce_op, shift)
    savefig(plt, joinpath(RESULT_DIR, "pathfinder_$ndists.pdf"))
    savefig(plt, joinpath(RESULT_DIR, "pathfinder_$ndists.png"))
end

# # No deflation

# optimizer2 = Optim.LBFGS(; m=6, linesearch=LineSearches.MoreThuente())
# q2, ϕ2, component_ids1 = multipathfinder(
#     logπ, ∇logπ, x₀s, ndraws; ndraws_elbo=100, ndraws_per_run=ndraws_per_run, optimizer=optimizer2,
#     rng=rng,
# )
# logπ3 = x -> logpdf(q2, [x])
# Plots.plot!(logπ3, minimum(means) - 5, maximum(means) + 5)
 