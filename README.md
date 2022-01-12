# deflation_examples

Please use the package environment by `julia --project=.` and `]instantiate`

## Truss TopOpt examples

```julia
using Runner
@runit "examples/truss_to/run.jl --problem dense_graph --task min_compliance_vol_constrained_deflation --optimizer mma"
```

## Continuum TopOpt examples

```julia
using Runner
@runit "examples\\cont_to\\run.jl --task min_vol_stress_constrained_deflation --optimizer percival"
```

Other cases:

```julia
@runit "examples\\cont_to\\run.jl --task min_compliance_vol_constrained --optimizer mma"
@runit "examples\\cont_to\\run.jl --task min_vol_compliance_constrained --optimizer mma"
```

## TODO
- stress-constrained continuum

- vol-constrained compliance-min truss
- buckling constrained truss
- mixed_integer truss