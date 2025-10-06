module Plot
using LinearAlgebra, Interpolations, Printf, Makie
export plot_solutions, error_grid

function plot_solutions(u,t, labels; l=1, normalise=true, hide_y=true, autolimits=true, steps = 32)
    x_steps = size(u, 1)
    x = range(0,l,length=x_steps)
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

function error_grid(fsolve, ref, fu0, dxs, dts)
	x_ref = range(0,1; length=size(ref,1))
	i_ref = [linear_interpolation(x_ref, u) for u in eachcol(ref)]
	err = Matrix{Float64}(undef, length(dts), length(dxs))
	for (j,dx) in enumerate(dxs)
		u0 = fu0(dx)
		x = range(0,1; length=size(u0,1))
		u_ref = stack(r.(x) for r in i_ref) 
		for (i,dt) in enumerate(dts)
			@printf("dx = %.2f, dt = %.2f\n", dx,dt)
			u = fsolve(u0,dx,dt)
			err[i,j] = maximum_error(u,u_ref)
		end
	end
	err
end

maximum_error(u,u_ref) = maximum(abs.(u.-u_ref))
maximum_rel_error(u,u_ref; epsilon = 1e-4) = maximum([abs(u-u_ref)/u_ref for (u,u_ref) in zip(u,u_ref) if u_ref>epsilon])
mean_error(u,u_ref) = mean(abs.(u.-u_ref))
mean_rel_error(u,u_ref; epsilon = 1e-4) = mean(maximum([abs(u-u_ref)/u_ref for (u,u_ref) in zip(u,u_ref) if u_ref>epsilon]))

end
