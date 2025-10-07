using LinearAlgebra
using Distributions
using Clustering
using JSON
using Parameters
# Set a seed for reproducibility
using Random
Random.seed!(25);

include("car-t_model.jl") # Model to fit

# const a_syms = [:ALPPL2, :MCAM]
const Lx  = 5
const Ly  = 5
const Δx  = 0.2
const Δy  = 0.2
const La  = 1.0
const Δa  = 0.2
const Nx  = Int(Lx / Δx)
const Ny  = Int(Ly / Δy)
const a_vals = 0.0:Δa:La
const Na  = length(a_vals)        # Number of discretised antigen density bins
# const Nva = Na^length(a_syms)+1   # Total number of discretised antigen density bins

const tspan = (0.0, 5.0) # days


n_size_glob = 1e8


include("ics_distr.jl")   # For setting initial conditions from distributions

function run_control(dists::NamedTuple)
    # Arbitrary CAR for output. Won't be expressed
    output_car = CAR(:control_CAR, [keys(dists)[1]])
    circuit = Circuit(control(output_car))
    run_trial(dists, circuit, T_size=0.0)
end

function run_constitutive_trial(dists::NamedTuple;
                                target_a_sym::Symbol = keys(dists)[1])
    output_car = CAR(:const_CAR, [target_a_sym])
    circuit = Circuit(constitutive(output_car))
    run_trial(dists, circuit)
end

function run_AND_trial(dists::NamedTuple;
                       priming_a_sym::Symbol = keys(dists)[1],
                       target_a_sym::Symbol = keys(dists)[2])
    # Define synNotch AND circuit
    output_car = CAR(:output_CAR, [priming_a_sym])
    circuit = Circuit(
        synNotch(output_car, priming_a_sym)
    )

    # Run trial for AND circuit
    run_trial(dists, circuit)
end

function run_trial(dists::NamedTuple, circuit::Circuit;
                   a_syms = keys(dists),
                   a_H = (; zip(a_syms, fill(0.0, length(a_syms)) )... ),
                   priming_a_sym::Symbol = a_syms[1],
                   target_a_sym::Symbol = a_syms[2],
                   n_size = n_size_glob, T_size = 1e6, n_H_size = 0,
                   # Model parameters
                   p = (
                       D_n      = 1e-4,          # Tumour diffusion (cm² day⁻¹) [1e-5, 10e-4]
                       r        = 0.21,          # Tumour proliferation (day⁻¹)
                       N_max    = 2.39e8*Δx*Δy,  # Tumour carrying capacity (cells cm⁻³)
                       k_m      = 1.5e-7,        # Mutation factor
                       e        = 0.41/Δx/Δy,    # Max killing rate (day⁻¹)
                       D_TA     = 0.0138,        # Active CAR T diffusion (cm² day⁻¹)
                       r_TA     = 0.18/Δx/Δy,          # Active CAR T proliferation (day⁻¹)
                       K_r      = 638*Δx*Δy,     # Saturation constant for CAR T proliferation (day⁻¹)
                       K_A      = 1808*Δx*Δy,    # CAR T recruitment saturation constant (cells cm⁻³)
                       k_A      = 0.65,          # CAR T cell activation rate
                       k_I      = 0.65,          # CAR T cell inhibition rate
                       k_O      = 0.65,          # CAR T cell apoptosis rate
                       l_TA     = 0.35,          # Active CAR T decay (day⁻¹)
                       D_TN     = 0.0138,        # Inactive CAR T diffusion (cm² day⁻¹)
                       l_TN     = 3e-5,          # Inactive decay (day⁻¹)
                   ), stochasticity = true)
    if (!(priming_a_sym in a_syms) ||
        !(target_a_sym in a_syms) ||
        length(a_H) != length(a_syms))
        error("Invalid AND trial inputs")
    end
    # tspan = (0.0, 6.0)            # Simulation time (days)
    Lx, Ly = 5, 5                 # Grid size (cm)
    Δx, Δy = 0.2, 0.2             # Grid discretisation step (cm)
    La = 1.0                      # Max antigen expression value. Normalised so 1.0 (a.u. ant.)
    Δa = 0.25                     # Antigen profile discretisation step, i.e. bin size (a.u. ant.)

    variances = (
        D_n      = 0.001,
        r        = 0.5,
        N_max    = 0.001,
        k_m      = 0.1,
        e        = 0.25,
        D_TA     = 0.1,
        r_TA     = 0.1,
        K_r      = 0.2,
        K_A      = 0.3,
        k_A      = 0.2,
        k_I      = 0.2,
        k_O      = 0.2,
        l_TA     = 0.2,
        D_TN     = 0.05,
        l_TN     = 0.2,
    )
    p = stochasticity ? (; [ k => v * rand(Uniform(1-variances[k], 1+variances[k]))
                             for (k,v) in pairs(p)]...) : p

    # Create model functions
    model = create_GatedCARTmodel(a_syms, circuit, Lx=Lx, Ly=Ly, La=La, Δx=Δx, Δy=Δy, Δa=Δa, a_H=a_H)

    Nx  = model.Nx         # Number of discretised grid cells, x
    Ny  = model.Ny         # Number of discretised grid cells, y    
    # Nva = model.Nva        # Total number of discretised antigen density bins
    Nva = Na^length(a_syms)+1
    N = Nx*Ny*Nva
    
    n0 = ics_distr(dists, Nx, Ny, Nva, Δa, n_size)
    n_H0 = n_H_size / Nx / Ny
    n_TN0 = T_size / Nx / Ny

    model = create_GatedCARTmodel(
        a_syms, circuit, Lx=Lx, Ly=Ly, La=La, Δx=Δx, Δy=Δy, Δa=Δa, a_H=a_H
    )
    u0 = model.initial_conditions(n0, n_H0 = n_H0, n_TN0 = n_TN0)
    prob = ODEProblem(model.pdesys!, u0, tspan, p)
    sol = solve(prob, Tsit5(), progress=true, progress_steps=1, saveat=0.1, abstol=1e-4, reltol=1e-4)
    return sol
end

function soltodens(sol, numantigens)
    n_t = []
    Nva = Na^numantigens # + 1 ??? Only works when I don't have +1... but whatever I'm sure it's fine 
    N = Nx * Ny * Nva
    for i in 1:length(sol)
        u = sol[i]
        n = reshape(u[1:N], Nx, Ny, Nva)
        push!(n_t, sum(@view n[:,:,1:Nva-1]) * Δx * Δy)
    end
    return n_t
end

function run_AND_trials(dists_vec::Vector{T}; Nt=300) where T<:NamedTuple
    t_common = range(tspan[1], tspan[2], length=Nt)

    #prim = :ALPPL2
    #targ = :MCAM

    controlfn(dists) = run_control(dists)
    constfn(dists) = run_constitutive_trial(dists) # , target_a_sym=targ)
    andfn(dists) = run_AND_trial(dists) # , target_a_sym=targ, priming_a_sym=prim)

    pcontrol = plot_trial(dists_vec, controlfn, Nt, t_common, title="Control")
    pconst   = plot_trial(dists_vec, constfn, Nt, t_common, title="Constitutive")
    pand     = plot_trial(dists_vec, andfn, Nt, t_common, title="SynNotch")
    
    # Save fig
    isdir("results") || mkdir("results")
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

    p = plot(pcontrol, pconst, pand, layout=(1,3), ylims=(0.0, 500000), xlims=(0.0, 8.0), linecolor=:black, fillcolor=:grey, fillalpha=0.25, linewidth=2, xlabel="Time", ylabel="Tumour density", size=(1500, 500)); gui()
    savefig("results/trials-$(timestamp).png")
    return p
end

function plot_trial(dists_vec::Vector{T}, fn::Function, Nt::Integer, t_common;
                    title::String = "Tumour size vs time") where T<:NamedTuple
    dens = [
        begin
            sol = fn(dists)
            dens = soltodens(sol, length(dists))
            # Interpolate to common timesteps
            vals = sol.(t_common) |> x -> getindex.(x,1)
        end for dists in dists_vec ]
    meanv = mean(hcat(dens...), dims=2)[:] # calculate mean
    minv = minimum(hcat(dens...), dims=2)[:]
    maxv = maximum(hcat(dens...), dims=2)[:]
    lowerv = meanv .- minv
    upperv = maxv .- meanv
    ribbon = (lowerv, upperv)
    
    plot(t_common, meanv, ribbon=ribbon, title=title, size=(1500,500))
end

function examples()
    z = 1.96
    dists1 = (
        ALPPL2 = let σ = ((log(1e6)-log(1e4))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
        MCAM = let σ = ((log(10^6.2)-log(1e3))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
    )
    dists2 = (
        ALPPL2 = let σ = ((log(1e7)-log(1e5))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
        MCAM = let σ = ((log(10^6.4)-log(1e2))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
    )
    run_AND_trials([dists1, dists2, dists1, dists2, dists1, dists2, dists1, dists2])
end

function example_trial()
    z = 1.96
    dists = (
        ALPPL2 = let σ = ((log(1e6)-log(1e4))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
        MCAM = let σ = ((log(10^6.2)-log(1e3))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
    )
    sol = run_AND_trial(dists)
    dens = soltodens(sol, length(dists))

    # Plot and save fig
    isdir("results") || mkdir("results")
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    plot(sol.t, dens, title="Tumour size over time")
    savefig("results/trial-$(timestamp).png")
end

# Converts parsed json dictionary of parameters to 
function dicttodist(dict::AbstractDict)
    return dict["Non_zero_dist"] == "weibull" &&
        Weibull(dict["Params"]["shape"], dict["Params"]["scale"])
end

# Generates a named tuple of Distribution objects from JSON file definition
function jsontodists(filepath::AbstractString)
    dicts = JSON.parsefile(filepath)
    (; [ Symbol(k) => dicttodist(v) for (k,v) in pairs(dicts)]... )
end

function create_AND_dists(prim::Symbol, targ::Symbol, all_dists::NamedTuple)
    (; prim => all_dists[prim], targ => all_dists[prim] )
end


function run_AND_trials(filepath::AbstractString="./dists_imputed.json")
    all_dists = jsontodists(filepath) # Read antigen expression distributions
    numpatients = 10                  # 10 virtual patients

    # Repeat for each set of patient distributions
    patients = repeat([
        create_AND_dists(:CD24_CID44971, :CD63_CID44971, all_dists)
        create_AND_dists(:CD24_CID4495,  :EPCAM_CID4495, all_dists)
        create_AND_dists(:BST2_CID4066,  :CD24_CID4066,  all_dists)
        create_AND_dists(:BST2_CID3963,  :CD74_CID3963,  all_dists)
    ], numpatients)
    # Run trials, save graph
    n_size_glob = 1e6;
    p1 = run_AND_trials(patients)
    
    n_size_glob = 1e7;
    p2 = run_AND_trials(patients)
    
    n_size_glob = 1e8;
    p3 = run_AND_trials(patients)

    return [p1; p2; p3]

    plot(p1, linecolor=:red, fillcolor=:red, fillalpha=0.2);
    plot!(twinx(), p2, linecolor=:blue, fillcolor=:blue, fillalpha=0.2);
    plot!(twinx(), p3, linecolor=:purple, fillcolor=:purple, fillalpha=0.2);
  
    savefig("results/AAAA.png");
end
