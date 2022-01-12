##############################

wait_for_key(prompt) = (print(stdout, prompt); read(stdin, 1); nothing)

##############################
# Distances

function multi_bernoulli_kl_divergence(_p, _q)
    # https://math.stackexchange.com/questions/2604566/kl-divergence-between-two-multivariate-bernoulli-distribution
    eps_num = 1e-6
    p = clamp.(_p, eps_num, 1-eps_num)
    q = clamp.(_q, eps_num, 1-eps_num)
    return sum(p .* log.(p./q) + (1 .- p) .* log.((1 .- p)./(1 .- q))) 
end