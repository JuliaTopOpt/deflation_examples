using Revise, Distributions, StatsFuns, Plots, ProgressMeter, AdvancedVI, ForwardDiff, DiffResults, LinearAlgebra, Random, Flux, StableRNGs, Zygote, ChainRulesCore, JLD

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

function stochastic_vi(means, initθ, rng; power = 3.0, shift = 0.0, nsamples = 5, niters = 10000, max_barrier_multiple = 1000, quad_multiple = 0.01, nsolves = length(means), radius = 0.0, verbose=false)
    mixture = mixturef(means)
    logπ = x -> logpdf(mixture, x[1])
    # Plots.plot(logπ, minimum(means) - 5, maximum(means) + 5, label = "target")
    variational_objective = AdvancedVI.ELBO()

    diff_results = DiffResults.GradientResult(initθ)
    function deflation(x, y)
        if length(solutions) > 0
            d = zero(eltype(x))
            for sol in solutions
                d += (1/max(kldivergence(getq(x), getq(sol)) - radius, 0))^power + shift
            end
            if verbose
                @info "d = $(ForwardDiff.value(d)), y = $(ForwardDiff.value(y))"
            end
            return d - y
        else
            return return -one(eltype(x))
        end
    end

    alg = ADVI(nsamples, niters)
    chunk_size = length(initθ)
    solutions = []
    runtimes = []
    for deflate_iter in 1:nsolves
        st_time = time()
        θ = copy(initθ) #[0.1 * randn(); 1.0; 1.0]
        optimizer = AdvancedVI.DecayedADAGrad()
        step = 1
        prog = ProgressMeter.Progress(alg.max_iters, 1, "deflation iter#$(deflate_iter): inner SGD iters")
        while (step ≤ alg.max_iters)
            if verbose
                @info "θ = $θ"
            end

            # decaying coefficient approaching 0
            multiple = max(step / alg.max_iters * max_barrier_multiple, 1)
            if verbose
                @info "multiple = $multiple"
            end

            function f(θ_)
                return -variational_objective(
                    rng, alg, getq(θ_[1:end-1]), logπ,
                    alg.samples_per_step
                ) + (-log(max(-deflation(θ_[1:end-1], exp(θ_[end])), 1e-8)) + quad_multiple * exp(θ_[3])^2) / multiple
            end
            if verbose
                @info "f(θ) = $(f(θ))"
            end

            chunk = ForwardDiff.Chunk(min(length(θ), chunk_size))
            config = ForwardDiff.GradientConfig(f, θ, chunk)
            ForwardDiff.gradient!(diff_results, f, θ, config)
            ∇ = DiffResults.gradient(diff_results)
            if verbose
                @info "grad = $∇"
            end

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
        runtime = time() - st_time
        push!(runtimes, runtime)
    end

    # Optimal distributions
    dists = getq.(solutions)

    # PDF plots
    plt = Plots.plot(x -> exp(logπ(x)), minimum(means) - 5, maximum(means) + 5, 
        linestyle=:solid, linewidth = 4, linecolor = :black,
        legend = true, label = "target pdf", grid=false)
    map(enumerate(dists)) do (i, dist)
        Plots.plot!(plt, x -> exp(logpdf(dist, [x])), minimum(means) - 5, maximum(means) + 5, 
            label = "approx pdf $i", linewidth = 1)
    end

    # Minimization objective values
    vals = map(dists) do dist
        -variational_objective(rng, alg, dist, logπ, 1000)
    end
    return vals, runtimes, plt
end