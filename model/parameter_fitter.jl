using Turing
using StatsPlots
using LinearAlgebra
using Distributions
using Clustering
using Parameters
# Set a seed for reproducibility
using Random
Random.seed!(25);

include("car-t_model.jl") # Model to fit

const a_syms = [:ALPPL2, :MCAM]
const Lx  = 5
const Ly  = 5
const Δx  = 0.2
const Δy  = 0.2
const La  = 1.0
const Δa  = 0.2
const Nx  = Int(Lx / Δx)
const Ny  = Int(Ly / Δy)
const Na  = length(a_vals)        # Number of discretised antigen density bins
const Nva = Na^length(a_syms)+1   # Total number of discretised antigen density bins

@kwdef struct DataFitterData
    saveat::Vector{Float64}
    data::Vector{Float64}
    a_syms::Vector{Symbol}
    # a_H::NamedTuple
    circuit::Circuit
    # u0 = nothing
    tspan::Tuple = (0.0, 10.0) # Simulation time (days)
    Lx::Real = Lx
    Ly::Real = Ly       # Grid size (cm)
    Δx::Real = Δx
    Δy::Real = Δy   # Grid discretisation step (cm)
    La::Real = La            # Max antigen expression value. Normalised so 1.0 (a.u. ant.)
    Δa::Real = Δa           # Antigen profile discretisation step, i.e. bin size (a.u. ant.)
    name::Symbol = :I_am_a_car_t_stop_As_yet_I_have_no_name_stop
end

include("ics_distr.jl")   # For setting initial conditions from distributions

# Extracts a vector of tumour volumes from model solution
function soltovol(problem, solution)
    Nx  = problem.Nx;   Ny = problem.Ny;   Nva = problem.Nva;
    Δx  = problem.Δx;   Δy = problem.Δy;   Δa  = problem.Δa;

    [ @views let u = solution
         n = reshape(u[1:N], Nx, Ny, Nva)
         n_TA = reshape(u[N+1:2N], Nx, Ny, Nva)
         n_TN = reshape(u[2N+1:2N+Nx*Ny], Nx, Ny)
         
         # Sum over antigen dimension to get total densities
         ntot = sum(n[:,:,1:Nva-1], dims=3)[:,:,1]

         # Cluster to find tumour clusters for volume estimation
         clusters = dbscan(ntot, 0.1, 5)
         
         # Find cluster volumes
         clustervols = Dict{Int, Float64}()
         for cluster_id in unique(clusters.assignments)
             if cluster_id == 0
                 continue # skip noise
             end

             # Use area as volume, assuming units cells cm^-2
             clustervols[cluster_id] =
                 # Get area, radius, and volume (assuming spherical)
                 #let A = count(x -> x == cluster_id, clusters.assignments) * Δx * Δy
                     #r = sqrt(A/π)
                     #V = (4/3)*π*r^3
                     #V
                 #end
                 count(x -> x == cluster_id, clusters.assignments) * Δx * Δy
         end

         # Return total volume of clusters as element
         sum(values(clustervols))
      end
    for i in 1:length(solution) ]
end



@model function fitcart()
    # Parameter priors
    D_n   ~ Uniform(10e-5, 10e-4) # cm²day⁻¹
    r     = 0.21
    N_max = 2.39e8
    k_m   = 1.5e-7
    e     = 0.41
    K_g   ~ truncated(Normal(0, 10), 0, Inf)
    D_TA  = 0.0138
    r_TA  = 0.18
    K_r   ~ truncated(Normal(0, 637.64), 0, Inf)
    K_A   ~ truncated(Normal(0, 1808.0), 0, Inf)
    k_A   = 0.65
    k_I   ~ truncated(Normal(0, 1.0), 0, 2)
    k_O   ~ truncated(Normal(0, 1.0), 0, 2)
    l_TA  = 0.0293
    D_TN  = 0.0138
    l_TN  = 3e-5

    # Circuit parameters, assumed uniform across circuits for ease of estimation
    β     ~ Uniform(0.0, 1.0)
    ε     ~ truncated(Normal(0.0, 0.5), 0.0, 1.0)
    k     ~ Uniform(0.24, 2.4)

    # Applicable only to synNotch chaining, thus will need to be fitted separately
    #h     ~ Uniform(1.0, 3.0)
    #K_R   ~ Uniform(0.0, 5.0)

    # Error parameter 
    σ ~ InverseGamma(2, 3) 


    control(data::Vector) = DataFitterData(
        saveat  = [0.0, 7.0, 15.0, 21.0, 28.0, 35.0],
        data    = data,
        a_syms  = a_syms,
        circuit = Circuit(constitutive(CAR(:controlcar, []))))
    synnotch(data::Vector) = DataFitterData(
        saveat  = [0.0, 7.0, 15.0, 21.0, 28.0, 35.0],
        data    = data,
        a_syms  = a_syms,
        circuit = Circuit(synNotch(CAR(:outputcar, [:MCAM]), :ALPPL2, β=β, ε=ε, K_A=K_A, k=k)))
    constitutivealppl2(data::Vector) = DataFitterData(
        saveat  = [0.0, 6.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0],
        data    = data,
        a_syms  = a_syms,
        circuit = Circuit(constitutive(CAR(:outputcar, [:ALPPL2]), K_A=K_A, k=k)))
    constitutivemcam(data::Vector) = DataFitterData(
        saveat  = [0.0, 6.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0],
        data    = data,
        a_syms  = a_syms,
        circuit = Circuit(constitutive(CAR(:outputcar, [:MCAM]), K_A=K_A, k=k)))

    # Data values. I don't want to parse a spreadsheet so I'm just copying these in
    control_wt_data = [ [0, 50.52, 63.16, 83.21, 124.52, 142.79], [0, 28.11, 63.94, 63.33, 101.87, 101.67], [0, 41.15, 68.51, 125.67, 316.84, 692.57], [0, 35.52, 46.23, 105.65, 98.81, 221.52], [0, 54.31, 79.77, 58.72, 167.56, 241.54] ]
    control_ko_data = [ [0.0, 28.15, 44.59, 52.48, 130.99, 197.07], [0.0, 38.97, 54.69, 80.2, 114.49, 122.83], [0, 43.5, 52.13, 69.88, 215.1, 225.73], [0, 39.4, 49.73, 64.62, 106.85, 116.59], [0, 48.83, 67.69, 69.44, 117.75, 110.9] ]
    synnotch_wt_data = [ [0, 33.43, 55.78, 33.65, 12.39, 46.04], [0, 57.67, 71.75, 25.71, 0, 0], [0, 44.95, 53.87, 0, 0, 0], [0, 60.13, 51.56, 43.27, 110.67, 77.13], [0, 40.08, 58.14, 24.58, 26.96, 40.75] ]
    synnotch_ko_data = [ [0, 45.59, 45.64, 77.8, 104.06, 124.73], [0, 60.31, 49.62, 40.35, 76.98, 84.02], [0, 50.95, 35.7, 37.12, 43.89, 59.66], [0, 57.05, 62.27, 79.74, 156.74, 277.42], [0, 51.21, 29.69, 62.6, 89.92, 276.11] ]
    alppl2_data = [ [0, 51.93, 62.52, 44.09, 44.38, 92.38, 177.03, 285.46], [0, 49.44, 40.23, 28.6, 17.54, 56.34, 81.88, 187.08], [0, 62.38, 69.89, 51.92, 33.13, 32.18, 45.42, 45.42], [0, 39.6, 47.83, 19.69, 0, 0, 16.47, 28.22], [0, 39.67, 47.8, 24.15, 20.52, 49.6, 78.6, 79.94] ]
    mcam_data = [ [0, 46.33, 79.17, 130.87, 170.34, 277.35, 444.19, 607.96], [0, 39.62, 45.68, 76.51, 79.63, 136.79, 248.98, 371.72], [0, 37.52, 55.53, 104.09, 132.62, 153.87, 249.04, 413.78], [0, 64.59, 70.14, 34.38, 34.22, 32.63, 20.18, 20.88], [0, 51.38, 62.78, 46.7, 20.22, 19.92, 18.99, 30.94] ]
    # TODO: additional synnotch data

    dataFitterDatas = [
        # Untransduced WT experiments
        #[control(data) for data in control_wt_data]...
        # Untransduced KO experiments
        #[control(data) for data in control_ko_data]...
        # synNotch WT experiments
        #[synnotch(data) for data in synnotch_wt_data]...
        # synNotch KO experiments
        #[synnotch(data) for data in synnotch_ko_data]...
        # constitutivealppl2 experiments
        [constitutivealppl2(data) for data in alppl2_data]...,
        # constitutivemcam experiments
        [constitutivemcam(data) for data in mcam_data]...,
        # synotch experiments
        #TODO
        # control experiments
        #TODO
    ]

    # Define distributions (roughly approximated to log normal distributions)
    # based on Hyrenius-Wittsten supplementary data Fig S1
    z = 1.96
    dists = (
        ALPPL2 = let σ = ((log(1e6)-log(1e4))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
        MCAM = let σ = ((log(10^6.2)-log(1e3))/(2 * z))
            truncated( LogNormal(log(10^5.4)+σ^2, σ), 0.0, 1e7 )
        end,
    )
    # Create initial conditions
    n0 = ics_distr(dists, Nx, Ny, Nva, Δa, n_size, T_size)
    n_H0 = 1e8
    n_TN0 = 1e12

    a_H = (; ALPPL2 = 1e3/1e7, MCAM = 1e3/1e7)
    
    function createmodel(datafitterdata) 
        # Unpack variables
        Lx, Ly = datafitterdata.Lx, datafitterdata.Ly
        Δx, Δy = datafitterdata.Δx, datafitterdata.Δy
        La = datafitterdata.La
        Δa = datafitterdata.Δa
        a_syms = datafitterdata.a_syms
        # a[H] = datafitterdata.a_H

        create_GatedCARTmodel(
            a_syms, circuit, Lx=Lx, Ly=Ly, La=La, Δx=Δx, Δy=Δy, Δa=Δa, a_H=a_H)
    end

    # For each experiment data model
    for i in 1:length(dataFitterDatas)
        model = createmodel(dataFitterDatas[i])
        u0 = model.initial_conditions(n0, n_H0 = n_H0, n_TN0 = n_TN0)
        prob = (model.pdesys!, u0, dataFitterDatas[i].tspan, p)
        pred = solve(prob, BS3(), progress=true, progress_steps=1, saveat=dataFitterDatas[i].saveat, abstol=1e-4, reltol=1e-4)
        
        for j in 1:length(dataFitterDatas[i].data)
            dataFitterDatas[i].data[j] ~ Normal(soltovol(pred[i]), σ)
        end
    end


    # TODO:
    # - randomise distributions of initial populations based on choe data /
    # - parameter estimation
    # - add Hyrenius-Wittsten data /
    # - parameter estimation
    # - optimisation?
end


# data::Vector
# a_syms::Vector{Symbol}
# a_H::NamedTuple
# circuit::Circuit
# u0
# tspan = (0.0, 10.0) # Simulation time (days)
# Lx, Ly = 5, 5       # Grid size (cm)
# Δx, Δy = 0.2, 0.2   # Grid discretisation step (cm)
# La = 1.0            # Max antigen expression value. Normalised so 1.0 (a.u. ant.)
# Δa = 0.25           # Antigen profile discretisation step, i.e. bin size (a.u. ant.)
# name = :as_yet_I_have_no_name

function create_chains()
    model = fitcart()

    # Sample 3 independent chains with forward-mode automatic differentiation (default)
    chain = Turing.sample(model, NUTS(), MCMCSerial(), 1000, 3; progress=false)


    # Plot trajectories of 300 random samples
    plot(; legend=false)
    posterior_samples = Turing.sample(chain[[:α, :β, :γ, :δ]], 300; replace=false)
    for p in eachrow(Array(posterior_samples))
        sol_p = solve(prob, Tsit5(); p=p, saveat=0.1)
        plot!(soltovol(sol_p); alpha=0.1, color="#BBBBBB")
    end

    # Plot simulation and noisy observations.
    plot!(sol; color=[1 2], linewidth=1)
    scatter!(sol.t, odedata'; color=[1 2])

    
    return chain
end


create_chains()
