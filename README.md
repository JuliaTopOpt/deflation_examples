# deflation_examples

Please use the package environment by `julia --project=.` and `]instantiate`

## Classical variational inference

```julia
using Runner
@runit "examples/variational_inference/stochastic.jl
```

## Pathfinder algorithm

```julia
using Runner
@runit "examples/variational_inference/pathfinder.jl
```

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
using Runner
@runit "examples\\cont_to\\run.jl --task min_compliance_vol_constrained --optimizer mma"
@runit "examples\\cont_to\\run.jl --task min_vol_compliance_constrained --optimizer mma"
```
