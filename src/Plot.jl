module Plot
using LinearAlgebra, Printf, Makie
export plot_solutions, error_grid

function plot_solutions(u,t, labels; l=1, normalise=true, hide_y=true, autolimits=true, steps = 32)
    x_steps = size(u[1], 1)
    x = range(0,l,length=x_steps)
    u = cat(u...;dims=3)
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
	axislegend()
    display(fig)
end

function error_grid(problem, solve, dxspan, dtspan; params=Dict())
	err = Matrix{Float64}(undef, length(dtspan), length(dxspan))
	for (j,dx) in enumerate(dxspan)
		x,t,u_ref = solve_auto(problem, abstol=1e-9, reltol=1e-6, dx=dx, dt=dt)
		for (i,dt) in enumerate(dtspan)
			_,_,u = solve(problem; dx=dx, dt=dt, params...)
			err[i,j] = maximum_error(u[end],u_ref[end])
		end
	end
	err
end

maximum_error(u,u_ref) = maximum(abs.(u.-u_ref))
maximum_rel_error(u,u_ref; epsilon = 1e-4) = maximum([abs(u-u_ref)/u_ref for (u,u_ref) in zip(u,u_ref) if u_ref>epsilon])
mean_error(u,u_ref) = mean(abs.(u.-u_ref))
mean_rel_error(u,u_ref; epsilon = 1e-4) = mean(maximum([abs(u-u_ref)/u_ref for (u,u_ref) in zip(u,u_ref) if u_ref>epsilon]))

end
