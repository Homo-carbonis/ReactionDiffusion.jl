module Plot
export plot, interactive_plot
using ..Simulate
using ..Models
using LinearAlgebra: norm

using Printf: @sprintf
using Makie
using Observables

function plot(model, params; kwargs...)
    u,t=simulate(model,params; full_solution=true, kwargs...)
    plot(model, u,t)
end

function plot(model, u, t; normalise=true, hide_y=true, autolimits=true, kwargs...)
    labels = [string(s.f) for s in species(model)]
    x_steps = size(u, 1)
    x = range(0.0,1.0,length=x_steps)
	r = normalise ? norm.(eachslice(u, dims=(2,3))) : ones(size(u)[2:3])
	fig=Figure()
	ax = Axis(fig[1,1])
	hide_y && hideydecorations!(ax)
    sg = SliderGrid(fig[2,1], (label="t",range=eachindex(t), format=i->@sprintf("%.2f",t[i])))
    sl=sg.sliders[1]
	T = lift(i -> t[i], sl.value)
	U = [lift(i -> u[:,i]/r[i], sl.value) for (u,r) in zip(eachslice(u, dims=2), eachrow(r))]
	for (U,label) in zip(U,labels)
		lines!(ax,x,U, label=label)
	end
	autolimits && on(sl.value) do _
	    autolimits!(ax)
	end
	axislegend(ax)
    display(fig)
end

# TODO Refactor so this shares code with simulate and plot.
function interactive_plot(model, param_ranges; hide_y=true, num_verts=32, kwargs...)
    simulate_ = simulate(model; num_verts=num_verts, kwargs...)
    function f(vals...)
        params = Dict(k => x isa Int ? v[x] : x for ((k,v), x) in zip(param_ranges,vals))
        u,t = parameter_set(model,params) |> simulate_
        u
    end

    
	fig=Figure()
	ax = Axis(fig[1,1])
	hide_y && hideydecorations!(ax)

    param_ranges = sort(param_ranges)
    slider_specs = [(label=string(k), range = v isa AbstractRange ? v : 1:length(v)) for (k,v) in param_ranges]

    sg = SliderGrid(fig[1,2], slider_specs...)

    U = lift(f, (sl.value for sl in sg.sliders)...)
    U = throttle(1/120, U) # Limit update rate to 120Hz
    x = range(0,1,num_verts)
    labels = [string(s.f) for s in species(model)]
    for i in eachindex(eachcol(U[]))
        lines!(ax, x, lift(u -> u[:,i], U); label=labels[i])
    end
    on(U) do _
	    autolimits!(ax)
	end
    axislegend(ax)
    display(fig)
end
end