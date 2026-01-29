module Models
export Model, species, parameters, reaction_parameters, boundary_parameters, diffusion_parameters,
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
using Catalyst: numspecies, numparams, assemble_oderhs, @species, @parameters, @reaction_network, ExprValues, get_usexpr, get_psexpr, esc_dollars!, find_parameters_in_rate!, forbidden_symbol_check, DEFAULT_IV_SYM, default_t, setmetadata
using ..Util: subst, ensure_function
# TODO CHECK for unnecessary Num conversions! Alternatively add needed Num conversions (and remove from Turing.jl)
"""
    Model(reaction, diffusion)

An object containing a mathematical description of a reaction diffusion system to be simulated, independent of parameter values.

# Fields
- `reaction::ReactionSystem`
- `diffusion::DiffusionSystem`
"""
struct Model
    reaction
    diffusion
    inital_conditions
end


# Don't try to broadcast over a model.
Base.broadcastable(model::Model) = Ref(model)

# Model getters
# TODO Eliminate unused getters.
species(model::Model) = Catalyst.species(model.reaction)
parameters(model::Model) = union(reaction_parameters(model), diffusion_parameters(model))

reaction_parameters(model::Model) = Catalyst.parameters(model.reaction)
diffusion_parameters(model::Model) = union(get_variables(model.diffusion.domain_size), parameters.(model.diffusion.spatial_reactions)...)

reaction_rates(model) = assemble_oderhs(model.reaction, species(model))

diffusion_rates(model::Model, default=0.0) = [get(model.diffusion.rates, s, default) for s in species(model)]

inital_conditions(model::Model, default=0.0) = [get(model.initial_conditions, s, default) for s in species(model)]



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
    S = species(model)
    R = reaction_rates(model)
    D = diffusion_rates(model)/L^2
    I = initial_conditions(model)
    pseudospectral_problem(S, R, D, I, num_verts; kwargs...)
end

ODESystem(model::Model) = convert(ODESystem, model.reaction)


struct DiffusionSystem
    domain_size
    rates
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

function diffusion_system(L, body, source)
    species,parameters,pairs = parse_body(body, source)
    L = parse_expr!(parameters, L)
    forbidden_symbol_check(parameters)
    psexpr = get_psexpr(parameters, Dict{Symbol, Expr}()) # @parameters
    sexpr = get_usexpr(species, Dict{Symbol, Expr}()) # @species
    dsexpr = :(DiffusionSystem($L, Dict($pairs)))
    quote
        $psexpr
        $sexpr
        $dsexpr
    end
end

macro initial_conditions(body)
    species,parameters,pairs = parse_body(body, source)
    psexpr = get_psexpr(parameters, Dict{Symbol, Expr}()) # @parameters
    sexpr = get_usexpr(species, Dict{Symbol, Expr}()) # @species
    icexpr = :(Dict($pairs))
    quote
        $psexpr
        $sexpr
        $icexpr
    end
end

function parse_body!(species,parameters, body, source)
    Base.remove_linenums!(body)
    parameters = ExprValues[]
    species = ExprValues[]
    pairs = Pair{ExprValues,ExprValues}[]

    for b in body.args
        r,s = b.args
        # Handle interpolation of variables
        r = parse_expr!(parameters,r)
        s = esc_dollars!(s)
        push!(pairs, s=>r)
        push!(species, s)
    end

    forbidden_symbol_check(species)
    forbidden_symbol_check(parameters)
    species, parameters, pairs
end

function parse_expr!(parameters, x)
    esc_dollars!(x)
    find_parameters_in_rate!(parameters, x)
    x
end


ParameterSet = Dict{Num, Float64}

"""
    function parameter_set(model, params; σ=0.001)

Create a set of parameter values and initial conditions for `model`.
Defaults are used for values missing from `params` and noise with standard deviation `σ` is added to the intial conditions. 
"""
function parameter_set(model, params; σ=0.001)
    set = ParameterSet()
    
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

end