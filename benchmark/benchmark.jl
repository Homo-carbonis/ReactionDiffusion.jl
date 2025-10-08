using DifferentialEquations, BenchmarkTools




"Binary search to find dx and dt needed to achieve the given tolerances"
function search(a,b, cost; max_steps=10)
    isze      cost(a,b)
    pass && return (dx,dt)

    function tune(dx,dt,steps)
        @show (dx,dt)
        steps == 0 && error("Failed to achieve tolerances.")
        dx_next = dx/2; dt_next = dt/2
        int_dx, transform_dx = init(dx_next,dt)
        int_dt, transform_dt = init(dx,dt_next)
        int_ref_dx = init_ref(dx_next)
        int_ref_dt = init_ref(dt_next)

        pass_dx,t_fail_dx = test_tol(int_dx, int_ref_dx, abstol, reltol; transform=transform_dx)
        pass_dt,t_fail_dt = test_tol(int_dt, init_ref_dt, abstol, reltol; transform=transform_dt)
        @show (t_fail_dx,t_fail_dt)

        if pass_dx
            dx_next, dt
        elseif pass_dt
            dx, dt_next
        elseif t_fail_dx > t_fail_dt
            tune(dx_next,dt,steps-1)
        else
            tune(dx,dt_next,steps-1)
        end
	end

    tune(dx,dt,max_steps)
end

"Binary search to find dx and dt needed to achieve the given tolerances"
function tune(init, init_ref, dx, dt; abstol=1e-4, reltol=1e-2, max_steps=10)
    pass,t_fail = test_tol(init(dx,dt), init_ref(dx), abstol, reltol)
    pass && return (dx,dt)

    function tune(dx,dt,steps)
        @show (dx,dt)
        steps == 0 && error("Failed to achieve tolerances.")
        dx_next = dx/2; dt_next = dt/2
        int_dx, transform_dx = init(dx_next,dt)
        int_dt, transform_dt = init(dx,dt_next)
        int_ref_dx = init_ref(dx_next)
        int_ref_dt = init_ref(dt_next)

        pass_dx,t_fail_dx = test_tol(int_dx, int_ref_dx, abstol, reltol; transform=transform_dx)
        pass_dt,t_fail_dt = test_tol(int_dt, init_ref_dt, abstol, reltol; transform=transform_dt)
        @show (t_fail_dx,t_fail_dt)

        if pass_dx
            dx_next, dt
        elseif pass_dt
            dx, dt_next
        elseif t_fail_dx > t_fail_dt
            tune(dx_next,dt,steps-1)
        else
            tune(dx,dt_next,steps-1)
        end
	end

    tune(dx,dt,max_steps)
end

function test_tol(int, int_ref, abstol, reltol; transform=identity)
    for _ in int
        @show int.t
        step!(int_ref, int.t, true)
        u = transform(int.u)
        u_ref = int_ref.u
        Plot.maximum_error(u, u_ref) < abstol && maximum_rel_error(u, u_ref) < reltol || return false, int.t
    end
    return true,int.t
end