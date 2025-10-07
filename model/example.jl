include("car-t_model.jl")    # Provides create_function(...)


function run_example_sim()  
    ######################################################################################
    # Parameters
    ######################################################################################
    # Simulation parameters
    tspan = (0.0, 10.0)           # Simulation time (days)
    
    Lx, Ly = 5, 5                 # Grid size (cm)
    Δx, Δy = 0.2, 0.2             # Grid discretisation step (cm)
    La = 1.0                      # Max antigen expression value. Normalised so 1.0 (a.u. ant.)
    Δa = 0.25                     # Antigen profile discretisation step, i.e. bin size (a.u. ant.)
    
    a_syms = (:ALPPL2, :MCAM, :EXC, :EXB, :EXA)
    a_H = (ALPPL2 = 0.0, MCAM = 0.0, EXC=0.0, EXB = 0.0, EXA = 0.0)    # Healthy tissue antigen profile. Order must be the same as a_syms

    # Define example circuit
    # synNotch1: priming antigen EXB, output tanCAR(MCAM, EXA)
    # synNotch2: priming antigen EXC, output transcription factor which controls synNotch1
    output_car = CAR(:output_CAR, [:MCAM, :EXA])
    tf         = TF(:tf1)
    circuit = Circuit(
        synNotch(output_car, :EXB, tf=tf),  
        synNotch(tf,         :EXC)
    )
    
    # Model parameters
    p = (
        D_n      = 1e-4,          # Tumour diffusion (cm² day⁻¹) [1e-5, 10e-4]
        r        = 0.21,          # Tumour proliferation (day⁻¹)
        N_max    = 2.39e8,        # Tumour carrying capacity (cells cm⁻³)
        k_m      = 1.5e-7,        # Mutation factor
        e        = 0.41,          # Max killing rate (day⁻¹)
        D_TA     = 0.0138,        # Active CAR T diffusion (cm² day⁻¹)
        r_TA     = 0.18,          # Active CAR T proliferation (day⁻¹)
        K_r      = 638,           # Saturation constant for CAR T proliferation (day⁻¹)
        K_A      = 1808,          # CAR T recruitment saturation constant (cells cm⁻³)
        k_A      = 0.65,          # CAR T cell activation rate
        k_I      = 0.65,          # CAR T cell inhibition rate
        k_O      = 0.65,          # CAR T cell apoptosis rate
        l_TA     = 0.35,          # Active CAR T decay (day⁻¹)
        D_TN     = 0.0138,        # Inactive CAR T diffusion (cm² day⁻¹)
        l_TN     = 3e-5,          # Inactive decay (day⁻¹)
    )

    # Create model functions
    model = create_GatedCARTmodel(a_syms, circuit, Lx=Lx, Ly=Ly, La=La, Δx=Δx, Δy=Δy, Δa=Δa, a_H=a_H)

    Nx  = model.Nx         # Number of discretised grid cells, x
    Ny  = model.Ny         # Number of discretised grid cells, y    
    Nva = model.Nva        # Total number of discretised antigen density bins

    # Initialise gradiated population from both antigens 0.0, to both antigens 1.0 as square in centre
    n0 = let T = Vector{Pair{NamedTuple{(:ALPPL2, :MCAM, :EXC, :EXB, :EXA), Tuple{Float64, Float64, Float64, Float64, Float64}}, Float64}}
        Matrix{T}(undef, Nx, Ny)
    end
    let tumour_size = 3, active_size = 2*tumour_size - 1, startx = Nx÷2 - 2, starty = Ny÷2 - 2
        for ix in 1:Nx, iy in 1:Ny
            if abs(ix - Nx÷2) < tumour_size && abs(iy - Ny÷2) < tumour_size
                # Gradient 
                gradx = (ix - startx) / (active_size - 1)
                grady = (iy - starty) / (active_size - 1)
                n0[ix, iy] = [ (ALPPL2 = gradx, MCAM = grady, EXC=gradx*grady, EXB=gradx*grady, EXA=gradx*grady) => 1e3 ]
            else
                n0[ix, iy] = []
            end
        end
    end
    
    ######################################################################################
    # Run simulation
    ######################################################################################
    u0 = model.initial_conditions(n0, n_H0 = 1e8, n_TN0 = 1e12) # Saturate with T cells
    prob = ODEProblem(model.pdesys!, u0, tspan, p)              # Define ODE problem for solving

    @show any(isnan, u0)             # Check for NaN values
    @show minimum(u0), maximum(u0)   # SHow minimum and maximum values

    # Solve ODE problem
    sol = solve(prob, BS3(), progress=true, progress_steps=1, saveat=1.0,
                abstol=1e-4, reltol=1e-4)

    ######################################################################################
    # Animate results
    ######################################################################################
    function get_vars(sol)
        n_t = []
        nTA_t = []
        nTN_t = []
        nH_t = []
        nTAH_t = []

        n_total_t = []

        N = Nx*Ny*Nva
        Δx, Δy = model.Δx, model.Δy
        
        for i in 1:length(sol)
            u = sol[i]
            n    = reshape(u[1:N], Nx, Ny, Nva)
            n_TA = reshape(u[N+1:2N], Nx, Ny, Nva)
            n_TN = reshape(u[2N+1:2N+Nx*Ny], Nx, Ny)
            # Sum over antigen dimension to get spatial 2D arrays
            push!(n_t,    @view sum(n[:,:,1:Nva-1], dims=3)[:, :, 1])
            push!(nTA_t,  @view sum(n_TA[:,:,1:Nva-1], dims=3)[:, :, 1])
            push!(nTN_t,  n_TN)

            # Sum over spatial dimension to get tumour volume, total T-cell pop
            push!(n_total_t, sum(n[:,:,1:Nva-1]) * Δx * Δy)
            
            # Healthy tissue related populations
            push!(nH_t,    @view n[:,:,Nva])
            push!(nTAH_t,  @view n_TA[:,:,Nva])
        end
        return n_t, nTA_t, nTN_t, nH_t, nTAH_t, n_total_t
    end

    n_t, nTA_t, nTN_t, nH_t, nTAH_t, n_total_t = get_vars(sol)

    # Get maximum values for colour scales
    maxn     = maximum([maximum(n) for n in n_t])
    maxn_TA  = maximum([maximum(n_TA) for n_TA in nTA_t])
    maxn_TN  = maximum([maximum(n_TN) for n_TN in nTN_t])
    maxn_H   = maximum([maximum(n) for n in nH_t])
    maxn_TAH = maximum([maximum(n_TA) for n_TA in nTAH_t])
    
    # Generate animated heatmap
    anim = @animate for i in 1:length(n_t)
        plotfn(var, title, clim) = heatmap(var[i], title=title, colorbar=true, clim=clim,
                                           aspect_ratio=1)
        p1 = plotfn(n_t, "Tumour n", (0.0,maxn))
        p3 = plotfn(nTA_t, "Active CAR T", :auto)
        p4 = plotfn(nTN_t, "Inactive CAR T", :auto)
        p5 = plotfn(nH_t, "Healthy tissue n", (0.0,maxn_H))
        p7 = plotfn(nTAH_t, "Healthy tissue activated CAR T", :auto)
        plot(p1, p3, p4, p5, p7, layout=(2,4), size=(1500,1000), titlefontsize=12)
    end

    # Create results directory if it doesn't exist
    isdir("results") || mkdir("results")

    # Generate timestamp string in format YYYYMMDD_HHMMSS
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")

    # Save GIF with timestamped filename
    gif(anim, "results/$(timestamp).gif", fps=2)

    # Plot total tumour population over time
    plot(sol.t, n_total_t, title="Tumour volume (number of cells) over time")
    savefig("results/$(timestamp).png")
end

run_example_sim()
