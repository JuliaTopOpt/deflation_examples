# deflation_examples

Please use the package environment by `julia --project=.` and `]instantiate`.

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
@runit "examples/truss_to/run.jl --problem dense_graph --task min_compliance_vol_constrained_deflation --optimizer nlopt --deflate_iters 5"
```

## Continuum TopOpt examples

```julia
using Runner
@runit "examples\\cont_to\\run.jl --task min_compliance_vol_constrained_deflation --optimizer nlopt --deflate_iters 5"
```
