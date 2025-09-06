function get_grid_periodic(N, min, max)
    full_range = range(min, max, length=N + 1)
    return range(full_range[1], stop=full_range[end-1],
        length=N)
end

function get_grid(N, min, max)
    return  range(min, max, length=N) 
end



function create_cache(; backend, d, vx, vy, vz, gridx, gridy, gridz, r1, t1, r2, ε)

    Nx, Ny, Nz = length(gridx), length(gridy), length(gridz)

    dx, dy, dz = step(gridx), step(gridy), step(gridz)
    dx_inv, dy_inv, dz_inv = 1 / dx, 1 / dy, 1 / dz

    xc = (gridx[end] - gridx[1]) / 2
    yc = (gridy[end] - gridy[1]) / 2

    ϕ0_val = ϕ0(gridz[end])

    d = adapt(backend, d)
    vx = adapt(backend, vx)
    vy = adapt(backend, vy)
    vz = adapt(backend, vz)
    gridx = adapt(backend, gridx)
    gridy = adapt(backend, gridy)
    gridz = adapt(backend, gridz)

    cache = (; backend, d, vx, vy, vz, gridx, gridy, gridz, Nx, Ny, Nz, xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv, dz, ϕ0_val)

    return cache
end
