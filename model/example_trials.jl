include("trial.jl")
p = run_AND_trials("./dists_imputed.json")

plot(p[1], linecolor=:red, fillcolor=:red, fillalpha=0.2);
plot!(twinx(), p[2], linecolor=:blue, fillcolor=:blue, fillalpha=0.2);
plot!(twinx(), p[3], linecolor=:purple, fillcolor=:purple, fillalpha=0.2);
savefig("results/AAAA.png");
