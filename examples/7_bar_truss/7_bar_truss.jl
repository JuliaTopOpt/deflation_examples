using LinearAlgebra, StatsFuns, Nonconvex
using LinearAlgebra: norm
Nonconvex.@load Ipopt
Nonconvex.@load NLopt
Nonconvex.@load Multistart

# Point load in the middle, kN
P = -100.0
# half-span, meter
h_span = 2.0

# O------C-------S
# \    / | \    /
#  \  /  |  \  /
#   \/   |   \/
#   X--------X'
# O = (0,0), X = (x,y) the variables
# norm(O,S) = 2*h_span
# P is a verticle point load applied at C
# X' node is mirrored

x0 = [1.0, -1.0]

function obj(X)
    x = X[1:2]
    # x = (x,y) of the variable node
    # compute the load path as a proxy for necessary structural weight
    L = [h_span, norm(x), norm(x-[h_span,0.0]), abs(h_span-x[1])*2]
    cosθ, sinθ = x/L[2]
    cosϕ, sinϕ = (x-[h_span,0.0])/L[3]
    F2 = 0.5*P/sinθ
    F3 = -0.5*P/sinϕ
    F = [-F2*cosθ, F2, F3, F2*cosθ+F3*cosϕ]
    FL = abs.(F).*L
    return 2*sum(FL[1:3]) + FL[4]
end

xstar = copy(x0)
shift = 1.0
power = 2.0
function deflation_constr(X)
    return 1.0/norm(X[1:2]-xstar, power) + shift - X[3]
end

model = Model(obj)
# X = (x1,y1,y), where y is the deflation auxilary variable
addvar!(model, [0.0, -3.0, -1e3], [h_span, 3.0, 1e3])
add_ineq_constraint!(model, deflation_constr)

alg = NLoptAlg(:LD_MMA)
options = NLoptOptions()
r = optimize(model, alg, vcat(x0, 1.0), options = options)

# alg = IpoptAlg()
# options = IpoptOptions(tol = 1e-4)
# r = optimize(model, alg, vcat(x0, 1.0), options = options)

println(r.minimizer, r.minimum)

# options = DeflatedOptions(
#     ndeflations = 10, sub_options = IpoptOptions(tol = 1e-4), radius = 2.0,
# )
# alg = DeflatedAlg(IpoptAlg())
# @time r = Nonconvex.optimize(model, alg, x0, options = options)

