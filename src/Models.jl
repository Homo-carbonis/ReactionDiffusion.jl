module Models
export Model, species, parameters, reaction_parameters, diffusion_parameters,
    num_species, num_params, num_reaction_params, num_diffusion_params,
    domain_size, initial_conditions, noise,
    reaction_rates, diffusion_rates,
    @diffusion_system,
    parameter_set, ParameterSet

import ..PseudoSpectral: pseudospectral_problem
export pseudospectral_problem

import ModelingToolkit: ODESystem
export ODESystem

import Random: seed!
export seed!

using Symbolics: Num, value
import Catalyst # Catalyst.species and Catalyst.parameters would conflict with our functions.
using Catalyst: numspecies, numparams, assemble_oderhs, @transport_reaction, @parameters
using ..Util: subst, ensure_function

"Contains all information about the model which is independent of the parameter values and method of solution."
struct Model
    reaction
    diffusion
    initial_conditions
    initial_noise
    seed
end

function Model(reaction, diffusion; initial_conditions=Dict(), initial_noise=0.01, seed=nothing)
    seed = something(seed, rand(Int))
    Model(reaction, diffusion, Dict(initial_conditions), initial_noise, seed)
end

# Don't try to broadcast over a model.
Base.broadcastable(model::Model) =  Ref(model)

# Model getters
# TODO Eliminate unused getters.
species(model::Model) = Catalyst.species(model.reaction)
parameters(model::Model) = union(reaction_parameters(model), diffusion_parameters(model))

reaction_parameters(model::Model) = Catalyst.parameters(model.reaction)
diffusion_parameters(model::Model) = union(Catalyst.parameters.(model.diffusion.spatial_reactions)...)

reaction_rates(model) = assemble_oderhs(model.reaction, species(model))
function diffusion_rates(model::Model, default=0.0)
    dict = Dict(r.species => r.rate for r in model.diffusion.spatial_reactions)
    Num.(subst(species(model), dict, default))
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
noise(model::Model) = model.initial_noise

reaction_parameters(model::Model, params, default=0.0) = subst(reaction_parameters(model), params, default)
#diffusion_parameters(model::Model, params, default=0.0) = get_vector(params, diffusion_parameters(model), default)

function diffusion_rates(model::Model, params::Dict{Symbol, Float64}, default=0.0) # wrong and bad
    syms = Dict(nameof(p) => p for p in parameters(model))
    params = Dict(syms[k] => v for (k,v) in params)
    [(substitute(D, params)) for D in diffusion_rates(model,default)]
end

initial_conditions(model::Model, default=0.0) = subst(species(model), model.initial_conditions, default)
pseudospectral_problem(model, num_verts; kwargs...) = pseudospectral_problem(species(model), reaction_rates(model), diffusion_rates(model), num_verts; kwargs...)
ODESystem(model::Model) = convert(ODESystem, model.reaction)

seed!(model::Model) = seed!(model.seed)


struct DiffusionSystem
    domain_size
    spatial_reactions
end


"""
    @diffusion_system L begin D, S;... end
Define a spatial domain of length and a set of diffusion rates. Values can be either fixed numbers or parameter symbols.
- `L`: Length of the domain.
- `D`: Diffusion rate.
- `S`: Species name.
"""
macro diffusion_system(L, body)
    diffusion_system(L,body,__source__)
end

macro diffusion_system(body)
    diffusion_system(1,body,__source__)
end

#TODO stop using transport_reaction.
function diffusion_system(L, body::Expr, source)
    Base.remove_linenums!(body)
    ps_expr = Expr(:(=), esc(L), Expr(:ref, Expr(:macrocall, Symbol("@parameters"), source, L), 1))
    trs_expr = Expr(:vect, (:(@transport_reaction $D/$L^2 $S) for (D,S) in getproperty.(body.args,:args))...)
    ds_expr = Expr(:call, DiffusionSystem, L, trs_expr)
    L isa Symbol ? Expr(:block, ps_expr, ds_expr) : ds_expr
end

ParameterSet = Dict{Num, Union{Float64,Function}}
function parameter_set(model, params; σ=0.001)
    params =  Dict(only(@parameters $k) => v for (k,v) in params) 
    set = ParameterSet()
    for s in species(model)
        p = get(params, s, 0.0)
        set[s] = iszero(σ) ? p : addnoise(σ) ∘ ensure_function(p)
    end

    for rs in reaction_parameters(model)
        set[rs] = get(params, rs, 1.0)
    end

    for ds in diffusion_parameters(model)
        set[ds] = get(params, ds, 0.0)
    end
    
    L = domain_size(model)
    if L isa Num 
        set[L] = get(params, L, 1.0)
    end
    set
end


addnoise(σ=1.0) = x -> max(0.0, x + σ*randn())

end