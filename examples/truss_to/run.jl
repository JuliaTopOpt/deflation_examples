using Makie
using TopOpt.TrussTopOptProblems.TrussVisualization: visualize

include("problem_defs.jl")
using .DeflateTruss

function optimize_truss()
    force = [0,-100.0]
    p = CustomPointLoadCantileverTruss((40,10),Tuple(ones(2)),1.0,0.3)
    visualize(p)
end

#########################################

using ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--opt2", "-o"
            help = "another option with an argument"
            arg_type = Int
            default = 0
        "--flag1"
            help = "an option without argument, i.e. a flag"
            action = :store_true
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    println("="^10)

    optimize_truss()
end

main()