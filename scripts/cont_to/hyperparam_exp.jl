using DrWatson
quickactivate(@__DIR__)
using UnPack
using ProgressMeter

include(srcdir("cont_to.jl"))

general_args = Dict(
    "problem_name"    => "half_mbb_beam",
    "opt_task"        => "min_compliance_vol_constrained",
    "mso_type"        => "deflation",
    "optimizer"       => "nlopt", 
    "distance"        => "l2",
    "deflation_iters" => 10, 
    "power"           => Array(3.0:1:6.), 
    "radius"          => Array(1.0:5:32), 
    "size_ratio"      => 2, 
    "verbose"         => false, 
    "replot"          => false,
)
arg_dicts = dict_list(general_args)

pbar = Progress(length(arg_dicts), 1)
for (i, args) in enumerate(arg_dicts)
    @unpack problem_name, opt_task, mso_type, optimizer, size_ratio = args
    @unpack distance, deflation_iters, power, radius = args
    problem_config = @ntuple problem_name opt_task mso_type optimizer distance deflation_iters power radius size_ratio 

    problem_result_dir = datadir("cont_to", "hyperparam_tests", savename(problem_config))
    args["problem_result_dir"] = problem_result_dir

    result_data =  optimize_domain(args)
    safesave(datadir("cont_to", "hyperparam_tests", savename(problem_config, "jld2")), result_data)
    ProgressMeter.next!(pbar)
end