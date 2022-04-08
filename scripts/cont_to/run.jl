using DrWatson # YES
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
            default = "min_compliance_vol_constrained_deflation" # "min_vol_stress_constrained_deflation"
        "--optimizer"
            help = "optimizer choice"
            arg_type = String
            default = "nlopt" # mma
        "--distance"
            help = "distance measure choice"
            arg_type = String
            default = "l2" # kl, w2
        "--deflation_iters"
            help = "print extra info."
            arg_type = Int
            default = 5
        "--verbose"
            help = "print extra info."
            action = :store_true
        "--write"
            help = "export result."
            action = :store_true
        "--replot"
            help = "parse and replot saved results."
            action = :store_true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    println(parsed_args)
    println("="^10)

    optimize_domain(parsed_args)
    # ["problem"], parsed_args["task"], verbose=parsed_args["verbose"],
    # write=parsed_args["write"], optimizer=parsed_args["optimizer"], distance=parsed_args["distance"],
    # deflation_iters=parsed_args["deflate_iters"], replot=parsed_args["replot"])
end

main()