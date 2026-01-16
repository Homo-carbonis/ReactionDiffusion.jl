module Models
export Model, species, parameters, reaction_parameters, diffusion_parameters,
    num_species, num_params, num_reaction_params, num_diffusion_params,
    domain_size, initial_conditions, noise,
    reaction_rates, diffusion_rates,
    @diffusion_system,
    parameter_set, ParameterSet

import ..PseudoSpectral: pseudospectral_problem
export pseudospectral_problem

import ModelingToolkit # Needed for Catalyst internal DSL macros
import ModelingToolkit: ODESystem
export ODESystem

using Symbolics: Num, value, get_variables
import Catalyst # Catalyst.species and Catalyst.parameters would conflict with our functions.
using Catalyst: numspecies, numparams, assemble_oderhs, @species, @parameters, ExprValues, get_usexpr, get_psexpr, esc_dollars!, find_parameters_in_rate!, forbidden_symbol_check, DEFAULT_IV_SYM, default_t, setmetadata
using ..Util: subst, ensure_function
# TODO CHECK for unnecessary Num conversions! Alternatively add needed Num conversions (and remove from Turing.jl)
"""
    Model(reaction, diffusion)

An object containing a mathematical description of a reaction diffusion system to be simulated, independent of parameter values, initial conditions and means of solution.

# Fields
- `reaction::ReactionSystem`
- `diffusion::DiffusionSystem`
"""
struct Model
    reaction
    diffusion
end

# Don't try to broadcast over a model.
Base.broadcastable(model::Model) =  Ref(model)

# Model getters
# TODO Eliminate unused getters.
species(model::Model) = Catalyst.species(model.reaction)
parameters(model::Model) = union(reaction_parameters(model), diffusion_parameters(model))

reaction_parameters(model::Model) = Catalyst.parameters(model.reaction)
diffusion_parameters(model::Model) = union(get_variables(model.diffusion.domain_size), diffusion_parameters.(model.diffusion.spatial_reactions)...)
boundary_parameters(model::Model) = union(boundary_parameters.(model.diffusion.spatial_reactions)...)

reaction_rates(model) = assemble_oderhs(model.reaction, species(model))

function diffusion_rates(model::Model, default=0.0)
    dict = Dict(r.species => r.rate for r in model.diffusion.spatial_reactions)
    subst(species(model), dict, default)
end

function boundary_conditions(model::Model, default=(0.0,0.0))
    dict = Dict(r.species => r.boundary_conditions for r in model.diffusion.spatial_reactions)
    subst(species(model), dict, default)
end

num_species(model::Model) = numspecies(model.reaction)
num_params(model::Model) = num_reaction_params(model) + num_diffusion_params(model)
num_reaction_params(model::Model) = numparams(model.reaction)
num_diffusion_params(model::Model) = length(diffusion_parameters(model))

domain_size(model::Model) = model.diffusion.domain_size
function domain_size(model::Model, params)
    L = domain_size(model)
    L isa Num ? params[nameof(L)] : L
end

is_fixed_size(model::Model) = typeof(domain_size(model)) != Num # TODO use type system. 

reaction_parameters(model::Model, params, default=0.0) = subst(reaction_parameters(model), params, default)
#diffusion_parameters(model::Model, params, default=0.0) = get_vector(params, diffusion_parameters(model), default)

function diffusion_rates(model::Model, params::Dict{Symbol, Float64}, default=0.0) # wrong and bad
    syms = Dict(nameof(p) => p for p in parameters(model))
    params = Dict(syms[k] => v for (k,v) in params)
    [(substitute(D, params)) for D in diffusion_rates(model,default)]
end

function pseudospectral_problem(model, num_verts; kwargs...)
    L = domain_size(model)
    pseudospectral_problem(species(model), reaction_rates(model), diffusion_rates(model)/L^2, (./).(boundary_conditions(model), L), num_verts; kwargs...)
end

ODESystem(model::Model) = convert(ODESystem, model.reaction)


struct DiffusionSystem
    domain_size
    spatial_reactions
end

struct SpatialReaction
    rate
    boundary_conditions
    species
end

"""
    @diffusion_system L begin D, [(a,b)], species;... end

Define a spatial domain of length `L` and a set of diffusion rates and Neumann boundary conditions for the given species.
The boundary conditions `uₓ(0)=a` and `uₓ(L)=b` default to 0 if ommitted.

# Example
```
@diffusion_system L begin
    0.5,             U
    Dᵥ, (0.0, 0.5),  V
    Dᵣ/k, (a, a*s),  R
end
```
"""
macro diffusion_system(L, body)
    diffusion_system(L,body,__source__)
end

macro diffusion_system(body)
    diffusion_system(1,body,__source__)
end


function diffusion_system(L, body::Expr, source)
    Base.remove_linenums!(body)
    parameters = ExprValues[]
    species = ExprValues[]
 
    # Build a vector of spatial_reaction constructor calls and simultaneously
    # collect parameter and species names.
    srexprs = [spatial_reaction(species, parameters, b.args...) for b in body.args]
    find_parameters_in_rate!(parameters, L)
    iv = :($(DEFAULT_IV_SYM) = default_t())
    
    forbidden_symbol_check(species)
    forbidden_symbol_check(parameters)

    sexprs = get_usexpr(species, Dict{Symbol, Expr}()) # @species
    pexprs = get_psexpr(parameters, Dict{Symbol, Expr}()) # @parameters
    dsexpr = :(DiffusionSystem($L, [$(srexprs...)]))

    quote
        $iv
        $sexprs
        $pexprs
        $dsexpr
    end
end

# Use zero-flux BCs by default.
spatial_reaction(species, parameters, rateex, s) = spatial_reaction(species, parameters, rateex, :(0.0,0.0), s)

function spatial_reaction(species, parameters, rateex, bcex, s)
    # Handle interpolation of variables
    rateex = esc_dollars!(rateex)
    bcex = esc_dollars!(bcex)
    s = esc_dollars!(s)
    push!(species, s)

    # Parses input expression.
    find_parameters_in_rate!(parameters, rateex)
    find_parameters_in_rate!(parameters, bcex.args[1])
    find_parameters_in_rate!(parameters, bcex.args[2])
    # Checks for input errors.

    # Creates expressions corresponding to actual code from the internal DSL representation.
    :(SpatialReaction($rateex,  $bcex, $s))
end

species(sr::SpatialReaction) = sr.species
diffusion_parameters(sr::SpatialReaction) = get_variables(sr.rate)
boundary_parameters(sr::SpatialReaction) = union(get_variables.(sr.boundary_conditions)...)


ParameterSet = Dict{Num, Union{Float64,Function}}

"""
    function parameter_set(model, params; σ=0.001)

Create a set of parameter values and initial conditions for `model`.
Defaults are used for values missing from `params` and noise with standard deviation `σ` is added to the intial conditions. 
"""
function parameter_set(model, params; σ=0.001)
    set = ParameterSet()
    
    # Initial conditions
    for s in species(model)
        p = get(params, nameof(s.f), 0.0)
        set[s] = iszero(σ) ? p : addnoise(σ) ∘ ensure_function(p)
    end

    for rs in reaction_parameters(model)
        set[rs] = get(params, nameof(rs), 1.0)
    end

    for ds in diffusion_parameters(model)
        set[ds] = get(params, nameof(ds), 0.0)
    end
    
    for ds in boundary_parameters(model)
        set[ds] = get(params, nameof(ds), 0.0)
    end
    
    # Domain size
    if !is_fixed_size(model)
        L = domain_size(model)
        set[L] = get(params, nameof(L), 1.0)
    end

    set
end

parameter_set(params::ParameterSet) = params


addnoise(σ=1.0) = x -> max(0.0, x + σ*randn())

end