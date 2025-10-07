using Base.Threads
using CoherentNoise
using Dates
using DifferentialEquations
using Logging: global_logger
using LoopVectorization
using Parameters
using Plots
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

include("circuit_model.jl")    # Provides create_function(...)

struct GatedCARTmodel
    Lx::Real  # Grid size (x)
    Ly::Real  # Grid size (y)
    La::Real  # Max antigen value
    Δx::Real  # Grid discretisation step size (x)
    Δy::Real  # Grid discretisation step size (y)
    Δa::Real  # Antigen discretisation step size
    Nx::Real  # Size of grid x in number of bins (= Lx ÷ Δx)
    Ny::Real  # Size of grid y in number of bins (= Ly ÷ Δy)
    Na::Real  # Size of antigen dimension per antigen in number of bins
    Nva::Real # Size of antigen dimension collapsed over all antigens
    a_syms::Tuple{Vararg{Symbol}} # Tuple of antigen names
    circuit::Circuit
    a_H::NamedTuple               # Healthy tissue antigen profile  
    initial_conditions::Function
    pdesys!::Function
end

"""
Returns a PDE system function and initial conditions function for the given arguments
in a GatedCARTmodel struct
All antigen profile values must be in the same order as a_syms
"""
function create_GatedCARTmodel(a_syms::Tuple{Vararg{Symbol}}, circuit::Circuit;
                               Lx=5.0, Ly=5.0, La=1.0, Δx=0.1, Δy=0.1, Δa=0.2, a_H=(; zip(a_syms, zeros(length(a_syms)))...) )::GatedCARTmodel
    #println("Threads: $(Threads.nthreads())")
    Threads.nthreads() >= 4 || @warn "Running with low thread count."

    # Calculate values used for indexing
    Nx = Int(Lx/Δx)            # Number of discretised grid cells, x
    Ny = Int(Ly/Δy)            # Number of discretised grid cells, y    
    a_vals = 0.0:Δa:La         # Antigen bin values
    Na = length(a_vals)        # Number of discretised antigen density bins
    Nva= Na^length(a_syms)+1   # Total number of discretised antigen density bins

    # Precompute all antigen value vectors
    # E.g (5,5,5) for 3 antigens, 5 bins each
    shape = ntuple(_ -> Na, length(a_syms))

    # A vector where e.g. a_vec[1] gives the vector of a_1 values
    a_vec = let a_tumour_vec = [ [a_vals[Tuple(CartesianIndices(shape)[ia])[j]]
                                  for ia in 1:Nva-1] for j in 1:length(a_syms) ]
        [ vcat(a, collect(a_H)) for a in a_tumour_vec ] # Append a_H as last index
    end
    
    # Precompute all antigen NamedTuples. a_tuples[ia] gives current antigen profile
    # N.B. does not include a_H, which must be handled separately at index Nva
    a_tuples = [ (; zip(a_syms, vals)...)
                 for vals in Iterators.product((a_vals for _ in 1:length(a_syms))...) ]
    
    # Laplacian term helper function. 5-point stencil finite difference method
    # https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Finite_differences
    # https://en.wikipedia.org/wiki/Five-point_stencil#In_two_dimensions
    function ∇²(A, dx, dy)
        ΔA = zeros(size(A))

        # N.B. difference of index by 1 is equivalent to change in grid by Δx or Δy
        # e.g. A[3:end, 2:end-1] represents grid cells Δx away from A[2:end-1, 2:end-1]
        
        # ∇²A = δ²A/δ²x² + δ²A/δ²y²
        # Use @turbo SIMD to vectorise for performance
        @turbo for i in 2:Nx-1, j in 2:Ny-1
            ΔA[i,j] = (A[i+1,j] - 2A[i,j] + A[i-1,j]) / dx^2 +
                (A[i,j+1] - 2A[i,j] + A[i,j-1]) / dy^2
        end

        # No-flux boundary conditions
        ΔA[1,:]   .= ΔA[2,:]      # Left
        ΔA[end,:] .= ΔA[end-1,:]  # Right
        ΔA[:,1]   .= ΔA[:,2]      # Up
        ΔA[:,end] .= ΔA[:,end-1]  # Down
        return ΔA
    end

    # Define laplacian operator
    # ∇²(A) = Lap((Δx, Δy))(A) # This uses EquivariantOperators.Lap. Too slow so not used.
    ∇²(A) = ∇²(A, Δx, Δy)      # This uses the helper function. Fastest option. 

    # Define m(\vec{a}⁽ⁱ⁾, \vec{a}) as k_m for single discrete antigen loss, 0 otherwise
    # Likely more efficient to replace with compile time check, storing a boolean matrix
    is_single_discrete_aloss(a1, a2) = count(((x, y),) ->
        x == 0 && x < y, zip(Tuple(a1), Tuple(a2))) == 1
    m(a1, a2, k_m) = is_single_discrete_aloss(a1, a2) ? k_m : 0
    
    # Get total number of grid cells to iterate over
    N = Nx*Ny*Nva

    # Unpack circuit variables
    ceqs  =  (; (ceq.output.name => ceq.equation
                 for ceq in circuit.equations)... )
    cCARs =  (; (ceq.output.name => ceq.output.antigens
                 for ceq in filter(ceq -> isa(ceq.output, CAR),  ceqs))... )
    ciCARs = (; (ceq.output.name => ceq.output.antigens
                 for ceq in filter(ceq -> isa(ceq.output, iCAR), ceqs))... )
    cOFFs =  (; (ceq.output.name => ceq.output.antigens
                 for ceq in filter(ceq -> isa(ceq.output, OFF),  ceqs))... )
    cvars = let statevarsend = 2N+Nx*Ny # End of n, n_TA, n_TN variables
        NamedTuple{keys(ceqs)}(
            ( ((i-1)*N + 1) + statevarsend : (i*N) + statevarsend
              for (i, _) in enumerate(keys(ceqs)) )
        ) end

    # Define g(R) function
    function g(cu, ix, iy, ia, carset)
        isempty(carset) && return 0.0
        avec = ia == Nva ? a_H : a_tuples[ia]
        gsum = 0.0
        for (var, asyms) in pairs(carset)
            R = (cu[var])[ix, iy, ia]
            @inbounds @simd for asym in asyms
                gsum += logistic(avec[asym]) * mm(R, K_A=0.5)
            end
        end
        mm(gsum, K_A=0.5)
    end
    
    # PDE system function to be returned
    function pdesys!(du, u, p, t)

        # Parameters
        @unpack D_n, r, N_max, k_m, e, D_TA, r_TA, K_r, K_A, k_A, k_I, k_O, l_TA, D_TN, l_TN = p
        
        # Current state variables
        n    = reshape(u[1:N], Nx, Ny, Nva)
        n_TA = reshape(u[1N+1:2N], Nx, Ny, Nva)
        n_TN = reshape(u[2N+1:2N+Nx*Ny], Nx, Ny)

        # State variables of circuit 
        cu = (; (var => reshape(u[indices], Nx, Ny, Nva)
                 for (var, indices) in pairs(cvars))...)

        # Preallocate d/dt arrays
        dn = zeros(size(n))
        dn_TA = zeros(size(n_TA))
        dn_TN = zeros(size(n_TN))
        # Preallocate vector of circuit d/dt arrays
        cdu = (; (var => zeros(Nx, Ny, Nva) for var in keys(cvars))...)

        # Alias ia_H as Nva = index of a_H for readability
        ia_H = Nva

        # Calculate Laplacians
        ∇²n = similar(n)
        ∇²n_TA = similar(n_TA)
        @views @threads for ia in 1:Nva
            ∇²n[:,:,ia] = ∇²(n[:,:,ia])
            ∇²n_TA[:,:,ia] = ∇²(n_TA[:,:,ia])
        end
        ∇²n_TN = ∇²(n_TN)

        # For each (x,y):
        # Using @views to avoid copying arrays => avoid redundant memory allocations
        # Multithreaded for performance. Should be thread-safe... if not, we'll know...
        @views @threads for idx in 1:(Nx*Ny)
            threadid = Threads.threadid()
            ix, iy = fld(idx-1, Ny) + 1, mod(idx-1, Ny) + 1
            
            # Sum populations over antigen profiles
            N_total = sum(n[ix,iy,1:Nva-1])

            # T transfer value local to thread
            T_transfer = 0.0

            # For each antigen profile
            for ia in 1:Nva
                # Tumour and healthy tissue populations
                n_growth = r * (1 - N_total/N_max) * n[ix,iy,ia]
                if (ia == ia_H)
                    # Healthy
                    n_kill = e * mm(n[ix,iy,ia], K_A = K_A) * n[ix,iy,ia]
                    dn[ix,iy,ia] = D_n * ∇²n[ix,iy,ia] - n_kill
                else
                    # Tumour
                    n_kill = e * mm(n[ix,iy,ia], K_A = K_A) * n_TA[ix,iy,ia]
                    # e.g. a_tuples[ia] = (MCAM = 0.23, CD20 = 0.45)
                    n_aloss = sum(iai -> m(a_tuples[iai], a_tuples[ia], k_m) * n[ix,iy,iai], 1:Nva-1)
                    dn[ix,iy,ia] = D_n * ∇²n[ix,iy,ia] + n_growth + n_aloss - n_kill
                end

                # Circuit protein concentrations
                for var in keys(cvars)
                    (cdu[var])[ix,iy,ia] = let a = ((ia == ia_H) ? a_H : a_tuples[ia])
                        ceqs[var](cu, n, a, ix, iy, ia)
                    end
                end

                # Calculate transfer activation functions g(R)
                g_A = g(cu, ix, iy, ia, cCARs)
                g_I = g(cu, ix, iy, ia, ciCARs)
                g_O = g(cu, ix, iy, ia, cOFFs)
                
                # Calculate population transfers from protein concentrations
                T_mm = mm(n[ix,iy,ia], K_A=K_A)
                T_A = k_A * g_A * T_mm * n_TN[ix,iy]
                T_I = k_I * g_I * T_mm * n_TA[ix,iy,ia]
                T_O = k_O * g_O * T_mm
                T_OA = T_O * n_TA[ix,iy,ia]
                T_ON = T_O * n_TN[ix,iy]
                
                # Active CAR T-cell populations
                dn_TA[ix,iy,ia] = D_TA * ∇²n_TA[ix,iy,ia] + r_TA * mm(n[ix,iy,ia], K_A=K_r) * n_TA[ix,iy,ia] +
                    T_A - T_I - T_OA - l_TA * n_TA[ix,iy,ia]

                # Representing integral kernel of dn_TN/dt
                T_transfer += T_A - T_I + T_ON
            end

            # Inactive CAR T population
            dn_TN[ix,iy] = D_TN * ∇²n_TN[ix,iy] - T_transfer - l_TN * n_TN[ix,iy]
        end

        # Bound populations >= 0
        n .= max.(n, 0.0)
        n_TA .= max.(n_TA, 0.0)
        n_TN .= max.(n_TN, 0.0)
        
        # Update PDE system variables
        du[1:N] = vec(dn)
        du[N+1:2N] = vec(dn_TA)
        du[2N+1:2N+Nx*Ny] = vec(dn_TN)

        for var in keys(cvars)
            du[cvars[var]] = vec(cdu[var])
        end
    end

    """
    n0: a matrix of size (Nx x Ny). An element of the matrix is of the form:
    ```
    [
        ...
        (A=0.2, B=0.7) => 1e4
        (A=0.3, B=0.7) => 1e5
        (A=0.2, B=0.7) => 1e2
        ...
    ]
    ```
    Which is an array of pairs, where each pair associates an antigen profile
    with the population size at this (x,y) of cells expressing that antigen profile.

    n_H0: a matrix of size (Nx x Ny). An element of the matrix is a Real value, which is the population
    size of the healthy tumour at that position.

    n_TN0: a matrix of size (Nx x Ny). An element of the matrix is a Real value, which is the population
    size of the inactive T cells at that position. 

    Alternatively, passing a Real for n_H0 or n_TN0 will set a uniform population of the Real value.

    WARNING This function does not currently rigorously check input types or formats. Passing matrices of
    mismatched sizes or invalid antigen profiles may result in unexpected behaviour. 
    """
    function initial_conditions(n0::AbstractMatrix{<:AbstractVector{<:Pair{<:NamedTuple, <:Real}}};
                                n_H0 = 0.0, n_TN0 = 0.0)
        # Error if matrices have the wrong sizes
        if !all(m0 -> m0 isa AbstractMatrix ? size(m0) == (Nx, Ny) : true, [n0, n_H0, n_TN0])
            error("Mismatched initial condition population matrix sizes. Must be of size ($(Nx), $(Ny)) or be a scalar.")
        end

        function set_var!(v, v0, ix, iy)
            if v0 isa AbstractMatrix
                v[ix, iy] = v0[ix, iy]
            elseif v0 isa Real
                v[ix, iy] = v0
            else
                error("Initial conditions must be either a matrix or a real number.")
            end
        end
        
        n = zeros(Nx, Ny, Nva)
        n_TA = zeros(Nx, Ny, Nva)
        n_TN = zeros(Nx, Ny)
        
        cu = (; (var => zeros(Nx, Ny, Nva) for var in keys(cvars))...)

        # For each x,y
        @views @inbounds for ix in 1:Nx, iy in 1:Ny
            # For each pair of the element of n0
            for p in n0[ix,iy]
                a = first(p)   # Get antigen profile
                n_C = last(p)  # Get population size

                # Find the matching antigen profile index, and set/add to the population size
                for ia in 1:Nva-1, sym in a_syms
                    # Round to nearest Δa (N.B. tolerance Δa/2 ∴ bins cannot overlap)
                    if abs(a_tuples[ia][sym] - a[sym]) < Δa/2
                        n[ix,iy,ia] += n_C     # Add to tumour population bin
                    end
                end
            end
            
            set_var!(n_TN, n_TN0, ix, iy)      # Set inactive CAR T-cell population
            set_var!(n[:,:,Nva], n_H0, ix, iy) # Set healthy tissue population
        end

        vcat(vec(n), vec(n_TA), vec(n_TN), [vec(cu[var]) for var in keys(cvars)]...)
    end

    return GatedCARTmodel(Lx, Ly, La, Δx, Δy, Δa, Nx, Ny, Na, Nva, a_syms, circuit, a_H, initial_conditions, pdesys!)
end
