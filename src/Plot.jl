module Plot
using LinearAlgebra, Statistics, Interpolations, Printf, Makie, Base.Threads
export plot_solutions, errormap

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

"Generate a 2d map of error with respect to dx and dt."
function errormap(fsolve, x, u0, u1, dxs, dts)
	err = Matrix{Float64}(undef, length(dxs), length(dts))
	u0_itp = [cubic_spline_interpolation(x, u) for u in eachcol(u0)]
	u1_itp = [cubic_spline_interpolation(x, u) for u in eachcol(u1)]
	L = x[end]-x[1]
	@threads for i in eachindex(dxs)
		local dx = dxs[i]
		local x = 0:dx:L
		local u0 = stack(itp.(x) for itp in u0_itp) 
		local u1 = stack(itp.(x) for itp in u1_itp) 
		for (j,dt) in enumerate(dts)
			@printf("dx = %.2f, dt = %.2f\n", dx,dt)
			local u = fsolve(u0,dt)
			err[i,j] = mean_error(u,u1)
		end
	end
	err
end

maximum_error(u,u_ref) = maximum(abs.(u.-u_ref))
maximum_rel_error(u,u_ref; epsilon = 1e-4) = maximum([abs(u-u_ref)/u_ref for (u,u_ref) in zip(u,u_ref) if u_ref>epsilon])
mean_error(u,u_ref) = mean(abs.(u.-u_ref))
mean_rel_error(u,u_ref; epsilon = 1e-4) = mean(maximum([abs(u-u_ref)/u_ref for (u,u_ref) in zip(u,u_ref) if u_ref>epsilon]))

end
