rng = StableRNG(123)
Random.seed!(rng, 123)
means = 30 * randn(rng, 20)
initθ = [0.0, 5.0, 6.0]
nsolves = 10
power = 3.0
nsamples = 10
radius = 1.0
niters = 10000
verbose = false
vals1, runtimes = run_exp(means, initθ, rng; nsolves=nsolves, power=power, nsamples=nsamples, radius=radius, verbose=verbose)

# power = 3.0
# vals2 = run_exp(means, initθ, rng; nsolves, power, nsamples, radius)
RESULT_DIR = joinpath(@__DIR__, "results");
savefig(joinpath(RESULT_DIR, "variational_inference.pdf"))
save(joinpath(RESULT_DIR, "variational_inference_result.jld"), "objectives", vals1, "runtimes", runtimes)