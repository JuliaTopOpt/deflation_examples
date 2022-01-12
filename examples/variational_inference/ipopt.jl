using Revise, Distributions, StatsFuns, Plots, ProgressMeter, AdvancedVI, ForwardDiff, DiffResults, LinearAlgebra, Random, Flux, StableRNGs, Zygote, ChainRulesCore, Nonconvex
Nonconvex.@load Ipopt

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

rng = StableRNG(123)
Random.seed!(rng, 123)
means = 30 * randn(rng, 20)
initθ = [0.0, 10.0, 3.0]
nsolves = 10
power = 5.0

struct FDFunc <: Function
    f
end
(f::FDFunc)(x) = f.f(x)
function ChainRulesCore.rrule(f::FDFunc, x::AbstractVector)
    ∇ = ForwardDiff.gradient(f, x)
    return f(x), Δ -> (NoTangent(), Δ * ∇)
end

function run_exp_ipopt(means, initθ, rng; power = 3.0, shift = 1.0, nsamples = 1000, max_iter = 100, nsolves = length(means))
    mixture = mixturef(means)
    logπ = x -> logpdf(mixture, x[1])
    plot(logπ, minimum(means) - 5, maximum(means) + 5)
    variational_objective = AdvancedVI.ELBO()

    solutions = []
    function deflation(x, y)
        if length(solutions) > 0
            d = zero(eltype(x))
            for sol in solutions
                d += (1/kldivergence(getq(x), getq(sol)))^power + shift
            end
            @info "d = $d, y = $y"
            return d - y
        else
            return return -one(eltype(x))
        end
    end

    for _ in 1:nsolves
        θ = copy(initθ)
        function f(θ_)
            return -variational_objective(
                rng, ADVI(nsamples, 1000000), getq(θ_[1:end-1]), logπ,
                nsamples
            )
        end
        g(θ_) = deflation(θ_[1:end-1], exp(θ_[end]))
        
        oalg = IpoptAlg()
        options = IpoptOptions(; max_iter)
        model = Model(FDFunc(f))
        addvar!(model, fill(-Inf, 3), fill(Inf, 3))
        add_ineq_constraint!(model, FDFunc(g))
        r = Nonconvex.optimize(model, oalg, initθ; options)
        push!(solutions, r.minimizer[1:end-1])
    end

    # Optimal distributions
    dists = getq.(solutions)

    # PDF plots
    plt = plot(x -> exp(logπ(x)), minimum(means) - 5, maximum(means) + 5, legend = true)
    map(dists) do dist
        plot!(plt, x -> exp(logpdf(dist, [x])), minimum(means) - 5, maximum(means) + 5)
        display(plt)
    end

    # Minimization objective values
    vals = map(dists) do dist
        -variational_objective(rng, ADVI(nsamples, 1000000), dist, logπ, 1000)
    end
    return vals
end

rng = StableRNG(123)
Random.seed!(rng, 123)
means = 20 * randn(rng, 20)
initθ = [0.0, 5.0, 6.0]
nsolves = 10
power = 3.0
_nsamples = 1000
max_iter = 50

vals = run_exp_ipopt(means, initθ, rng; nsamples = _nsamples, nsolves, max_iter)
