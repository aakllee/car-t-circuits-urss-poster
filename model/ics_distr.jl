using Distributions

# Distribution is antigen expression levels associated with the cell
# densities expressing that level. So the vector of distributions is
# the population distributions for each antigen.
# `dists` is a named tuple associating antigens with their distributions
# `model` is the Gated CAR-T cell model struct
# Assumes antigen expressions are independent
function initial_conditions_matrix_from_distribution(dists::NamedTuple, Nx, Ny, Nva, Δa, n_size)
    N = Nx * Ny * Nva
    a_syms = keys(dists)

    # Find number of cells to assign per grid cell per sample
    magn = n_size / Nva / Nx / Ny # PLACEHOLDER value

    round_to_Δ(a, Δa) = round(a / Δa) * Δa
    # Scales sample [0,1]
    get_sample(dist::Distribution) = let v = rand(dist), # sample distribution
        min = minimum(dist), max = maximum(dist)    # get min/max for scaling
        round_to_Δ((v - min)/(max - min), Δa) # round to nearest antigen bin
    end

    # Sums cell counts of identical antigen profiles
    function sum_by_key(pairs::Vector{Pair{T, S}}) where {T<:NamedTuple, S<:Real}
        sums = Dict{NamedTuple, Real}()
        for (k, v) in pairs
            sums[k] = get(sums, k, 0) + v
        end
        return collect(sums)
    end

    # Initialise tumour population matrix
    n0 = let T = Vector{Pair{NamedTuple, Float64}}
        Matrix{T}(undef, Nx, Ny)
    end

    # For each grid square
    for ix in 1:Nx, iy in 1:Ny
        # Assign vector of tumour subpopulations
        n0[ix, iy] = sum_by_key([
            # Key is antigen profile, sampled from distribution
            let k = (; [a_sym => get_sample(dist) for (a_sym, dist) in pairs(dists)]...),
                # Value is population size. This is summed over all duplicate keys
                v = magn
                # Assign pair as element of vector
                (k => v)
            end
            # Repeat for number of antigen bins
            for _ in 1:Nva-1
                ])
    end

    n0
end
ics_distr = initial_conditions_matrix_from_distribution
