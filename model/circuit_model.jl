"""
(Lam 2024 https://pubs.acs.org/doi/10.1021/acssynbio.4c00230) 
Represents synNotch activation
"""
logistic(a::Real; β = 0.9, ε = 0.25) = 1 / (1 + exp(-(a-β)/ε))
mm(n::Real; K_A = 1000)             = n/(n + K_A)
hill(R::Real; h = 2, K_R = 0.5)      = R^h / (R^h + K_R^h)

function rbeq(n::Real, a::Real, R::Real; k = 0.1, K_A = 1000, β = 0.9, ε = 0.25)
    logistic(a, β=β, ε=ε) * mm(n, K_A=K_A) - k * R
end
function rbeq(n::Real, a::Real, R::Real, TF::Real; k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25)
    logistic(a, β=β, ε=ε) * mm(n, K_A=K_A) * hill(TF, h=h, K_R=K_R) - k * R
end

abstract type AbstractProtein end
struct CAR <: AbstractProtein
    name::Symbol             # CAR name symbol
    antigens::Vector{Symbol} # Vector of antigen names
end
struct TF <: AbstractProtein
    name::Symbol             # TF name
end
struct iCAR <: AbstractProtein
    name::Symbol
    antigens::Vector{Symbol} # Vector of antigen names
end
struct OFF <: AbstractProtein
    name::Symbol
    antigens::Vector{Symbol}
end

abstract type CircuitEquation end

struct BindingEquation <: CircuitEquation
    output::AbstractProtein
    equation::Function
end
# TODO: cleanse
function BindingEquation(output::AbstractProtein, binding_antigen::Symbol; tf::Union{Nothing, TF}=nothing,
                         k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25)
    if isnothing(tf) return BindingEquation(output, BindingEquationfn(output, binding_antigen,     k=k,K_A=K_A,β=β,ε=ε))
    else             return BindingEquation(output, BindingEquationfn(output, binding_antigen, tf, k=k,K_A=K_A,h=h,K_R=K_R,β=β,ε=ε))
    end
end
function BindingEquation(output::AbstractProtein, binding_antigen::Symbol, tf::TF;
                         k = 0.1, K_A=1000, β = 0.9, ε = 0.25)
    BindingEquation(output, BindingEquationfn(output, binding_antigen, tf,
                                              k=k,K_A=K_A,β=β,ε=ε))
end
function BindingEquationfn(output::AbstractProtein, binding_antigen::Symbol;
                           k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25)
    (cu, n, a_tuple, ix, iy, ia) -> rbeq(n[ix,iy,ia], a_tuple[binding_antigen],
                                         cu[output.name][ix,iy,ia],
                                         k=k,K_A=K_A,β=β,ε=ε)
end
function BindingEquationfn(output::AbstractProtein, binding_antigen::Symbol, tf::TF;
                           k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25)
    (cu, n, a_tuple, ix, iy, ia) -> rbeq(n[ix,iy,ia], a_tuple[binding_antigen],
                                         cu[output.name][ix,iy,ia], cu[tf.name][ix,iy,ia],
                                         k=k,K_A=K_A,h=h,K_R=K_R,β=β,ε=ε)
end

function synNotch(output::AbstractProtein, binding_antigen::Symbol; tf::Union{Nothing, TF}=nothing,
                  k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25)
    BindingEquation(output, binding_antigen, tf=tf, k=k,K_A=K_A,h=h,K_R=K_R,β=β,ε=ε)
end

struct ConstitutiveEquation <: CircuitEquation
    output::AbstractProtein
    equation::Function
end
function constitutive(output::AbstractProtein; k = 0.1, K_A=1000)
    ConstitutiveEquation(
        output, 
        (cu, n, a_tuple, ix, iy, ia) -> mm(n[ix,iy,ia], K_A=K_A) -
            k * cu[output.name][ix,iy,ia]
    )
end

struct ControlEquation <: CircuitEquation
    output::AbstractProtein
    equation::Function
end
function control(output::AbstractProtein; k=0.1, K_A=1000)
    ControlEquation(
        output,
        # Why not just set it to 0, instead of multiplying, I hear you ask?
        # Well, it breaks if I don't and I don't know why. So go figure.
        (cu, n, a_tuple, ix, iy, ia) -> (mm(n[ix,iy,ia], K_A=K_A) -
            k * cu[output.name][ix,iy,ia]) * 0.0
    )
end

# synNotch functions
#function synNotch(output::AbstractProtein, binding_antigen::Symbol; tf::TF=nothing,
#                  k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25) # TODO: default synNotch params
#    if isnothing(tf) return BindingEquation(output, BindingEquationfn(output, binding_antigen,     k=k,K_A=K_A,h=h,K_R=K_R,β=β,ε=ε))
#    else             return BindingEquation(output, BindingEquationfn(output, binding_antigen, tf, k=k,K_A=K_A,h=h,K_R=K_R,β=β,ε=ε))
#    end
#end
#function synNotch(output::AbstractProtein, binding_antigen::Symbol;
#                  k = 0.1, K_A=1000, β = 0.9, ε = 0.25)
#    BindingEquation(output, binding_antigen,
#                    k=k,K_A=K_A,β=β,ε=ε)
#end
#function synNotch(output::AbstractProtein, binding_antigen::Symbol, tf::TF;
#                  k = 0.1, K_A=1000, h = 2, K_R = 0.5, β = 0.9, ε = 0.25) 
#    BindingEquation(output, binding_antigen, tf,
#                    k=k,K_A=K_A,h=h,K_R=K_R,β=β,ε=ε)
#end

struct Circuit
    equations::Vector{CircuitEquation}
end
Circuit(ceqs::CircuitEquation...) = Circuit(collect(ceqs))
Circuit(ceq::CircuitEquation)     = Circuit([ceq])
Base.iterate(c::Circuit) = iterate(c.equations)
Base.iterate(c::Circuit, state) = iterate(c.equations, state)

function example()
    output_CAR = CAR(:output_CAR, [:MCAM, :EXB])
    tf_exa     = TF(:tf_exa)
    c = Circuit(
        BindingEquation(output_CAR, :EXC, tf = tf_exa),
        BindingEquation(tf_exa,     :EXA)
    ) 
end
