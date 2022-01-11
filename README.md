# deflation_examples

Please use the package environment by `julia --project=.` and `]instantiate`

## Truss TopOpt examples

You can either directly run from the command line: 
```bash
$ julia --project=. examples/truss_to/run.jl --opt compliance_deflation -v
```

Or using a running Julia session (recommended): 
```julia
julia> using Runner; @runit "examples/truss_to/run.jl -v --opt compliance_deflation"
```