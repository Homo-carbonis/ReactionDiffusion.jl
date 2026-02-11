using FFTW: r2r, REDFT00

Φ1(a,b,n) = [-2*(n-1)/sqrt(2*(n-1))*(b-a) ; zeros(n-1)]

DCT(u,n) = 1/sqrt(2*(n-1)) * r2r(u, FFTW.REDFT00)

function Φ2(a,b,n)
    u = fill(b-a,n)
    DCT(u,n)
end

function Φ3(a,b,n)
    X = range(0.0,1.0,n)
    ϕ = X.^2 * (b-a)/2 + X * a
    k = 0:n-1 # Wavenumbers
    h = 1/(n-1)
    σ² = @. -(4/h^2) * sin(k*pi/(2*(n-1)))^2
    σ² .* DCT(ϕ,n)
end


function Φ4(a,b,n)
    [0.0; ones(n-1)] * -2*(n-1)/sqrt(2*(n-1))*(b-a)
end