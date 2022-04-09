using DrWatson
quickactivate(@__DIR__)

include(srcdir("cont_to.jl"))

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    # ! @unpack problem_name, opt_task, verbose, write, optimizer, distance, deflation_iters, replot = args
    @add_arg_table! s begin
        "--problem_name"
            help = "problem to run"
            arg_type = String
            default = "half_mbb_beam"
        "--opt_task"
            help = "problem formulation"
            arg_type = String
            default = "min_compliance_vol_constrained" # "min_vol_stress_constrained"
        "--mso_type"
            help = "type of multi-solution optimization to run"
            arg_type = String
            default = "deflation" # "random_restart", "none"
        "--optimizer"
            help = "optimizer choice"
            arg_type = String
            default = "nlopt" # mma
        "--distance"
            help = "distance measure choice"
            arg_type = String
            default = "l2" # kl, w2
        "--deflation_iters"
            help = "number of iteration of deflation."
            arg_type = Int
            default = 5
        "--power"
            arg_type = Float64
            default = 4.0
        "--radius"
            arg_type = Float64
            default = 30.0
        "--size_ratio"
            help = "sizing the initial domain's dimension."
            arg_type = Int
            default = 2
        "--verbose"
            help = "print extra info."
            action = :store_true
        "--replot"
            help = "parse and replot saved results."
            action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    println("Parsed args:")
    println(args)
    println("="^10)

    @unpack problem_name, opt_task, mso_type, optimizer, size_ratio = args
    @unpack distance, deflation_iters, power, radius = args
    problem_config = @ntuple problem_name opt_task mso_type optimizer distance deflation_iters power radius size_ratio 

    problem_result_dir = datadir("cont_to", "attempts", savename(problem_config))
    args["problem_result_dir"] = problem_result_dir

    result_data =  optimize_domain(args)
    safesave(datadir("cont_to", "attempts", savename(problem_config, "jld2")), result_data)
end

main()