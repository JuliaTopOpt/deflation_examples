using Revise, Distributions, StatsFuns, Plots, ProgressMeter, AdvancedVI, ForwardDiff, DiffResults, LinearAlgebra, Random, Flux, StableRNGs, Zygote, ChainRulesCore

function mixturef(means, stds = ones(length(means)))
    return MixtureModel(Normal.(means, stds))
end

function kldivergence(p::MvNormal, q::MvNormal)
    # This is the generic implementation for AbstractMvNormal, you might need to specialize for your type
    length(p) == length(q) ||
        throw(DimensionMismatch("Distributions p and q have different dimensions $(length(p)) and $(length(q))"))
    # logdetcov is used separately from _cov for any potential optimization done there
    div = (tr(cov(q) \ cov(p)) + sqmahal(q, mean(p)) - length(p) + logdetcov(q) - logdetcov(p)) / 2
    return div
end
getq(θ) = MvNormal([θ[1]], Diagonal([exp(θ[2])]))

function run_exp(means, initθ, rng; power = 3.0, shift = 0.0, nsamples = 5, niters = 10000, max_barrier_multiple = 1000, quad_multiple = 0.01, nsolves = length(means), radius = 0.0)
    mixture = mixturef(means)
    logπ = x -> logpdf(mixture, x[1])
    plot(logπ, minimum(means) - 5, maximum(means) + 5, label = "target")
    variational_objective = AdvancedVI.ELBO()

    diff_results = DiffResults.GradientResult(initθ)
    function deflation(x, y)
        if length(solutions) > 0
            d = zero(eltype(x))
            for sol in solutions
                d += (1/max(kldivergence(getq(x), getq(sol)) - radius, 0))^power + shift
            end
            @info "d = $(ForwardDiff.value(d)), y = $(ForwardDiff.value(y))"
            return d - y
        else
            return return -one(eltype(x))
        end
    end

    alg = ADVI(nsamples, niters)
    chunk_size = length(initθ)
    solutions = []
    for _ in 1:nsolves
        θ = copy(initθ) #[0.1 * randn(); 1.0; 1.0]
        optimizer = AdvancedVI.DecayedADAGrad()
        step = 1
        prog = ProgressMeter.Progress(alg.max_iters, 1)
        while (step ≤ alg.max_iters)
            @info "θ = $θ"

            multiple = max(step / alg.max_iters * max_barrier_multiple, 1)
            @info "multiple = $multiple"

            function f(θ_)
                return -variational_objective(
                    rng, alg, getq(θ_[1:end-1]), logπ,
                    alg.samples_per_step
                ) + (-log(max(-deflation(θ_[1:end-1], exp(θ_[end])), 1e-8)) + quad_multiple * exp(θ_[3])^2) / multiple
            end
            @info "f(θ) = $(f(θ))"

            chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
            config = ForwardDiff.GradientConfig(f, θ, chunk)
            ForwardDiff.gradient!(diff_results, f, θ, config)
            ∇ = DiffResults.gradient(diff_results)
            @info "grad = $∇"

            if !all(isfinite, ∇)
                d = θ .* randn(rng, length(θ))
                @. θ = θ - d
                continue
            end
            Δ = AdvancedVI.apply!(optimizer, θ, ∇)
            all(isfinite, Δ) || continue
            @. θ = θ - Δ

            step += 1
            ProgressMeter.next!(prog)
        end
        push!(solutions, θ[1:end-1])
    end

    # Optimal distributions
    dists = getq.(solutions)

    # PDF plots
    plt = plot(x -> exp(logπ(x)), minimum(means) - 5, maximum(means) + 5, legend = true, label = "target pdf")
    map(enumerate(dists)) do (i, dist)
        plot!(plt, x -> exp(logpdf(dist, [x])), minimum(means) - 5, maximum(means) + 5, label = "approx pdf $i")
        display(plt)
    end

    # Minimization objective values
    vals = map(dists) do dist
        -variational_objective(rng, alg, dist, logπ, 1000)
    end
    return vals
end

rng = StableRNG(123)
Random.seed!(rng, 123)
means = 30 * randn(rng, 20)
initθ = [0.0, 5.0, 6.0]
nsolves = 10
power = 3.0
nsamples = 10
radius = 1.0
niters = 10000
vals1 = run_exp(means, initθ, rng; nsolves, power, nsamples, radius)

# power = 3.0
# vals2 = run_exp(means, initθ, rng; nsolves, power, nsamples, radius)
savefig("results/variational_inference.pdf")
