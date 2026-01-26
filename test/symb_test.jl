using Symbolics
n=5
@variables u[1:n]

x = range(0.0,1.0,n)
ϕ = [1*u[1],2,3,4,5*u[5]]
@show ϕ
_,fϕ! = build_function(u+ϕ, u; expression=Val{false})

u0=randn(n)
fϕ!(u0,u0)
u0