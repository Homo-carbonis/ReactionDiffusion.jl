module Util
using Pipe: @pipe
using Catalyst: @species, @parameters
using Symbolics: get_variables, Num

## Generic Dictionaries
"Convenience function to construct a dict using (k=v, ...) syntax"
dict(;kwargs...) = Dict(kwargs)

"Return a vector of dictionaries containing the cartesian product of the given values."
product(;kwargs...) = vec([Dict(zip(keys(kwargs), vals)) for vals in Iterators.product(values(kwargs)...)])
product(dict::Dict) = product(;dict...)

"Replace each element of keys with either the corresponding value in dict or default."
function subst(keys, dict, default)
    v = Vector(undef,0)
    for k in keys
        val = get(dict, k, default)
        push!(v, val)
    end
    v
end

"Zip a keys vector and a values vector into a dictionary."
zip_dict(keys, values) = Dict(zip(keys,values))
"Unzip a dictionary into a keys vector and a values vector."
unzip_dict(dict) = (keys(dict),values(dict))



## Parameters 
"Sort parameters by name."
sort_params(p) = sort(p, by=nameof)
"Replace parameter names with actual Symbolics variables."
lookup(name::Symbol) = only(@parameters $name)
lookup(param::Num) = param
"Extract parameters from a set of expressions and sort them by name."
collect_params(exprs, vars=[]) = @pipe exprs .|> get_variables |> splat(union) |> setdiff(_,vars) |> sort_params


# Parameter dictionaries
lookup(params::AbstractDict) = Dict(lookup(k) => v for (k,v) in params)
"Return a vector of parameter dictionaries with symbolic keys."
ensure_params_vector(params) = params |> ensure_vector .|> Dict .|> lookup

## Subscripted symbols
"Map integers to subscript characters."
sub(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))

"Subscript a symbol with `i...` separated by `_`."
subscript(X, i...) = Symbol(X, join(sub.(i), "_"))

"Build an array of subscripted symbols."
function subscripts(name, dim)
    dim = tuple(dim...) # ensure tuple
    ixs = Iterators.product(range.(1,dim)...)
    [subscript(name, r...) for r in ixs]
end

## Subscripted Catalyst variable arrays.
"Define a set of subscripted species and return them as a vector."
function defspecies(name, t, n)
    names = subscripts(name, n)
    [only(@species $n(t)) for n in names]
end
"Define a set of subscripted species and return them as a vector."
function defparams(name, dim)
    names = subscripts(name, dim)
    [only(@parameters $n) for n in names]
end
"Define a subscripted parameter."
function defparam(name, i...)
    name = subscript(name, i...)
    only(@parameters $name)
end

## Misc
issingle(x) = !(x isa AbstractVector)
isnonzero(x) = !(ismissing(x) || iszero(x))
ensure_vector(x) = x isa Vector ? x : [x]

end

