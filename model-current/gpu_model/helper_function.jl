function get_grid_periodic(N, min, max)
    full_range = range(min, max, length=N + 1)
    return range(full_range[1], stop=full_range[end-1],
        length=N)
end

function get_grid(N, min, max)
    return range(min, max, length=N)
end


function initial_condition(x, y, z, xc, yc, all_physical_parameters)
    (; r1, t1, r2, t2, ϕ0, df, uf, ϕs, dp) = all_physical_parameters
    r = sqrt((x - xc)^2 + (y - yc)^2)
    if r < r1 # inside inner pipe
        d_val = df
        vx_val = 0
        vy_val = 0
        vz_val = -uf
        ϕ_val = ϕs
    elseif r < r1 + t1 # inner pipe
        d_val = dp
        vx_val = 0
        vy_val = 0
        vz_val = 0
        ϕ_val = ϕs
    elseif r < r2 # between inner and outer pipe
        d_val = df
        vx_val = 0
        vy_val = 0
        vz_val = uf
        ϕ_val = ϕs
    elseif r < r2 + t2 # outer pipe
        d_val = dp
        vx_val = 0
        vy_val = 0
        vz_val = 0
        ϕ_val = ϕs
    else # rock
        d_val = dr
        vx_val = 0
        vy_val = 0
        vz_val = 0
        ϕ_val = ϕ0(z)
    end

    return d_val, vx_val, vy_val, vz_val, ϕ_val

end

function create_cache(; backend, d, vx, vy, vz, gridx, gridy, gridz, r1, t1, r2, ε)

    Nx, Ny, Nz = length(gridx), length(gridy), length(gridz)

    dx, dy, dz = step(gridx), step(gridy), step(gridz)
    dx_inv, dy_inv, dz_inv = 1 / dx, 1 / dy, 1 / dz

    xc = (gridx[end] - gridx[1]) / 2
    yc = (gridy[end] - gridy[1]) / 2

    ϕ0_val = ϕ0(gridz[end])

    idx_map_inner, idx_map_between = create_radius_index_maps(gridx, gridy, xc, yc, r1, t1, r2)
    N_inner = length(idx_map_inner)
    N_between = length(idx_map_between)


    d = adapt(backend, d)
    vx = adapt(backend, vx)
    vy = adapt(backend, vy)
    vz = adapt(backend, vz)
    gridx = adapt(backend, gridx)
    gridy = adapt(backend, gridy)
    gridz = adapt(backend, gridz)
    idx_map_inner = adapt(backend, idx_map_inner)
    idx_map_between = adapt(backend, idx_map_between)

    cache = (; backend, d, vx, vy, vz, gridx, gridy, gridz, Nx, Ny, Nz, xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv, dz, ϕ0_val, N_inner, N_between, idx_map_inner, idx_map_between)

    return cache
end


function create_radius_index_maps(gridx, gridy, xc, yc, r1, t1, r2)
    idx_map_inner = Tuple{Int,Int}[]
    idx_map_between = Tuple{Int,Int}[]
    for (i, x) in enumerate(gridx)
        for (j, y) in enumerate(gridy)
            r = sqrt((x - xc)^2 + (y - yc)^2)
            if r < r1 # inside inner pipe
                push!(idx_map_inner, (i, j))
            elseif r1 + t1 < r < r2 # between inner and outer pipe
                push!(idx_map_between, (i, j))
            end

        end
    end
    return idx_map_inner, idx_map_between
end


# kernels:


@kernel function kernel_rhs!(dϕ, @Const(ϕ), @Const(d), @Const(vx), @Const(vy), @Const(vz), @Const(gridx), @Const(gridy), xc, yc, r1, t1, r2, ε, dx_inv, dy_inv, dz_inv)
    i, j, k = @index(Global, NTuple)
    i, j, k = i + 1, j + 1, k + 1

    half = eltype(ϕ)(0.5)

    # Convective term
    x = gridx[i]
    y = gridy[j]
    r = sqrt((x - xc)^2 + (y - yc)^2)
    if r < r1 # inside inner pipe
        conv = (
            ε * vx[i, j, k] * (ϕ[i+1, j, k] - ϕ[i, j, k]) * dx_inv
            + ε * vy[i, j, k] * (ϕ[i, j+1, k] - ϕ[i, j, k]) * dy_inv
            + ε * vz[i, j, k] * (ϕ[i, j, k+1] - ϕ[i, j, k]) * dz_inv
        )
    elseif r1 + t1 < r < r2 # between inner and outer pipe 
        conv = (
            ε * vx[i, j, k] * (ϕ[i, j, k] - ϕ[i-1, j, k]) * dx_inv
            + ε * vy[i, j, k] * (ϕ[i, j, k] - ϕ[i, j-1, k]) * dy_inv
            + ε * vz[i, j, k] * (ϕ[i, j, k] - ϕ[i, j, k-1]) * dz_inv
        )
    else
        conv = 0
    end

    # Diffusive term
    diff = (
        (d[i+1, j, k] + d[i, j, k]) * half * (ϕ[i+1, j, k] - ϕ[i, j, k]) * dx_inv
        -
        (d[i, j, k] + d[i-1, j, k]) * half * (ϕ[i, j, k] - ϕ[i-1, j, k]) * dx_inv
    ) * dx_inv
    +
    (
        (d[i, j+1, k] + d[i, j, k]) * half * (ϕ[i, j+1, k] - ϕ[i, j, k]) * dy_inv
        -
        (d[i, j, k] + d[i, j-1, k]) * half * (ϕ[i, j, k] - ϕ[i, j-1, k]) * dy_inv
    ) * dy_inv
    +
    (
        (d[i, j, k+1] + d[i, j, k]) * half * (ϕ[i, j, k+1] - ϕ[i, j, k]) * dz_inv
        -
        (d[i, j, k] + d[i, j, k-1]) * half * (ϕ[i, j, k] - ϕ[i, j, k-1]) * dz_inv
    ) * dz_inv


    dϕ[i, j, k] = diff - conv

end

@kernel function kernel_boundary_x!(ϕ, Nx)
    j, k = @index(Global, NTuple)
    j, k = j + 1, k + 1

    # Do both x boundaries in one kernel
    ϕ[1, j, k] = ϕ[2, j, k]        # left boundary
    ϕ[Nx, j, k] = ϕ[Nx-1, j, k]    # right boundary
end

@kernel function kernel_boundary_y!(ϕ, Ny)
    i, k = @index(Global, NTuple)
    i, k = i + 1, k + 1

    # Do both y boundaries in one kernel
    ϕ[i, 1, k] = ϕ[i, 2, k]       # front boundary  
    ϕ[i, Ny, k] = ϕ[i, Ny-1, k]   # back boundary
end

@kernel function kernel_boundary_z!(ϕ, Nz, ϕ0_val)
    i, j = @index(Global, NTuple)
    i, j = i + 1, j + 1

    # Do both z boundaries in one kernel
    ϕ[i, j, 1] = ϕ[i, j, 2]       # top boundary
    ϕ[i, j, Nz] = ϕ0_val          # bottom boundary (Dirichlet)
end

# the following three kernels give only a slight speedup, but make it quite a bit more complicated
@kernel function kernel_diffusion!(dϕ, @Const(ϕ), @Const(d), dx_inv, dy_inv, dz_inv)

    i, j, k = @index(Global, NTuple)
    i, j, k = i + 1, j + 1, k + 1

    half = eltype(ϕ)(0.5)

    # Diffusive term
    diff = (
        (d[i+1, j, k] + d[i, j, k]) * half * (ϕ[i+1, j, k] - ϕ[i, j, k]) * dx_inv
        -
        (d[i, j, k] + d[i-1, j, k]) * half * (ϕ[i, j, k] - ϕ[i-1, j, k]) * dx_inv
    ) * dx_inv
    +
    (
        (d[i, j+1, k] + d[i, j, k]) * half * (ϕ[i, j+1, k] - ϕ[i, j, k]) * dy_inv
        -
        (d[i, j, k] + d[i, j-1, k]) * half * (ϕ[i, j, k] - ϕ[i, j-1, k]) * dy_inv
    ) * dy_inv
    +
    (
        (d[i, j, k+1] + d[i, j, k]) * half * (ϕ[i, j, k+1] - ϕ[i, j, k]) * dz_inv
        -
        (d[i, j, k] + d[i, j, k-1]) * half * (ϕ[i, j, k] - ϕ[i, j, k-1]) * dz_inv
    ) * dz_inv

    dϕ[i, j, k] = diff
end

@kernel function kernel_convection_inner!(dϕ, @Const(ϕ), @Const(vx), @Const(vy), @Const(vz), ε, dx_inv, dy_inv, dz_inv, idx_map_inner)
    ij, k = @index(Global, NTuple)
    i, j = idx_map_inner[ij]

    k = k + 1

    conv = (
        ε * vx[i, j, k] * (ϕ[i+1, j, k] - ϕ[i, j, k]) * dx_inv
        + ε * vy[i, j, k] * (ϕ[i, j+1, k] - ϕ[i, j, k]) * dy_inv
        + ε * vz[i, j, k] * (ϕ[i, j, k+1] - ϕ[i, j, k]) * dz_inv
    )
    dϕ[i, j, k] -= conv

end

@kernel function kernel_convection_between!(dϕ, @Const(ϕ), @Const(vx), @Const(vy), @Const(vz), ε, dx_inv, dy_inv, dz_inv, idx_map_between)
    ij, k = @index(Global, NTuple)
    i, j = idx_map_between[ij]

    k = k + 1

    conv = (
        ε * vx[i, j, k] * (ϕ[i, j, k] - ϕ[i-1, j, k]) * dx_inv
        + ε * vy[i, j, k] * (ϕ[i, j, k] - ϕ[i, j-1, k]) * dy_inv
        + ε * vz[i, j, k] * (ϕ[i, j, k] - ϕ[i, j, k-1]) * dz_inv
    )
    dϕ[i, j, k] -= conv

end



# callbacks
function save_julia_array(u, t, integrator)
    return copy(adapt(CPU(), u))
end

function save_julia_array_and_write_to_VTK_prepend_path(u, t, integrator, prepend_file)
    u_cpu = copy(adapt(CPU(), u))
    gridx = integrator.p.gridx
    gridy = integrator.p.gridy
    gridz = integrator.p.gridz
    file_name = prepend_file * "temperature"
    t = isinteger(t) ? Int(t) : t # make an int if possible for nicer file names
    save(path, file_name, u_cpu, gridx, gridy, gridz, t)
    return u_cpu
end

function save_and_print_callback(saveat; print_every_n=100, write_to_file=false, prepend_file="")
    # reset counter
    step_counter = Ref(0)
    # Callback that increments counter and prints every 100 steps
    function print_condition(u, t, integrator)
        step_counter[] += 1
        return step_counter[] % print_every_n == 0
    end

    function print_affect!(integrator)
        println("Step $(step_counter[]), t = $(integrator.t)")
    end

    # to have a process for long simulations
    print_cb = DiscreteCallback(print_condition, print_affect!)



    saved_values = SavedValues(Float64, Array{Float64,3})

    if write_to_file
        # closer of the other function to make it work with Callback Interface
        save_julia_array_and_write_to_VTK(u, t, integrator) = save_julia_array_and_write_to_VTK_prepend_path(u, t, integrator, prepend_file)
        save_cb = SavingCallback(save_julia_array_and_write_to_VTK, saved_values, saveat=saveat)
    else
        save_cb = SavingCallback(save_julia_array, saved_values, saveat=saveat)
    end



    return (CallbackSet(save_cb, print_cb), saved_values)

end



