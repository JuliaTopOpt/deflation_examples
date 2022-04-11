using DrWatson
quickactivate(@__DIR__)
using StableRNGs

include(srcdir("var_inf.jl"))

seed = 123
rng = StableRNG(seed)
Random.seed!(rng, seed)
mean_amp = 30
# mean_dim = 20
mean_dim = 5
means = mean_amp * randn(rng, mean_dim)
initθ = [0.0, 5.0, 6.0]
power = 3.0
nsamples = 10
radius = 1.0
niters = 10000
verbose = false

problem_config = @ntuple seed mean_amp mean_dim initθ power radius niters
args = @strdict rng mean_amp mean_dim initθ power radius niters

objectives, runtimes, plt = stochastic_vi(means, initθ, rng; 
    power=power, nsamples=nsamples, radius=radius, verbose=verbose)
result_data = merge(copy(args), @strdict objectives runtimes)

safesave(datadir("stochastic_vi", "attempts", savename(problem_config, "jld2")), result_data)
safesave(datadir("stochastic_vi", "attempts", savename(problem_config, ".pdf")), plt)
safesave(datadir("stochastic_vi", "attempts", savename(problem_config, ".png")), plt)
plt