using DrWatson

general_args = Dict(
    "problem_name"    => "half_mbb_beam",
    "opt_task"        => "min_compliance_vol_constrained_deflation",
    "verbose"         => false, 
    "write"           => false, 
    "optimizer"       => "nlopt", 
    "distance"        => "l2",
    "deflation_iters" => 5, 
    "replot"          => false,
)