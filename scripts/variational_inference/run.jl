using DrWatson
quickactivate(@__DIR__)

include(srcdir("var_inf.jl"))

rng = StableRNG(123)
Random.seed!(rng, 123)
mean_amp = 30
mean_dim = 20
means = mean_amp * randn(rng, mean_dim)
initθ = [0.0, 5.0, 6.0]
nsolves = 10
power = 3.0
nsamples = 10
radius = 1.0
niters = 10000
verbose = false

problem_config = @ntuple rng mean_amp mean_dim initθ nsolves power radius niters
args = @strdict rng mean_amp mean_dim initθ nsolves power radius niters

objectives, runtimes, plt = stochastic_vi(means, initθ, rng; 
    nsolves=nsolves, power=power, nsamples=nsamples, radius=radius, verbose=verbose)
result_data = merge(copy(args), @strdict objectives runtimes)

safesave(datadir("stochastic_vi", "attempts", savename(problem_config, "jld2")), result_data)
safesave(datadir("stochastic_vi", "attempts", savename(problem_config, ".pdf")), plt)
safesave(datadir("stochastic_vi", "attempts", savename(problem_config, ".png")), plt)
plt