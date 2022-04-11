using DrWatson
quickactivate(@__DIR__)
using UnPack
using ProgressMeter

include(srcdir("var_inf.jl"))

general_args = Dict(
    "seed"     => 123,
    "mean_amp" => 30,
    "mean_dim" => Array(10:1:20),
    "nsolves"  => 10,
    "nsamples" => 10,
    "niters"   => 10000,
    "power"    => 3.0,
    "radius"   => 1.0,
    "initθ"    => [0.0, 5.0, 6.0],
    "verbose"  => false,
)

arg_dicts = dict_list(general_args)

pbar = Progress(length(arg_dicts), 1, "VI multi-instance experiments...")
for (i, args) in enumerate(arg_dicts)
    @unpack seed, mean_amp, mean_dim, nsolves, nsamples, niters, power, radius = args
    println(args)

    rng = StableRNG(seed)
    Random.seed!(rng, seed)
    means = mean_amp * randn(rng, mean_dim)

    objectives, runtimes, plt = stochastic_vi(means, initθ, rng;
        nsolves=nsolves, power=power, niters=niters, nsamples=nsamples, radius=radius, 
        verbose=verbose)
    result_data = merge(copy(args), @strdict objectives runtimes)

    safesave(datadir("stochastic_vi", "expriments", savename(problem_config, "jld2")), result_data)
    safesave(datadir("stochastic_vi", "expriments", savename(problem_config, ".pdf")), plt)
    safesave(datadir("stochastic_vi", "expriments", savename(problem_config, ".png")), plt)
    ProgressMeter.next!(pbar)
end