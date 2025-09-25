function get_grid_periodic(N, min, max)
    full_range = range(min, max, length=N + 1)
    return range(full_range[1], stop=full_range[end-1],
        length=N)
end

function get_grid(N, min, max)
    return range(min, max, length=N)
end

function set_up_physics()
    # Physical parameters ##########################################################


    # Earth surface temperature [°C]
    ϕs = 20
    # Rock temperature as a function of depth [°C]
    ϕ0_open(d, ϕs) = ϕs + 0.5 * d
    ϕ0(d) = ϕ0_open(d, ϕs) # to make sure we dont work with a global variable

    # Rock: granite #########################################
    # Rock density [g/m3]
    ρr = 2750000
    # Rock specific heat [J/(g °C)]
    cr = 0.790
    # Rock thermal conductivity [W/(m °C)]
    λr = 2.62
    # Rock diffusion coefficient
    dr = λr / (ρr * cr)

    # Pipes: polyethylene ####################################
    # Inner pipe inside radius [m]
    r1 = 0.1
    # Inner pipe thickness [m]
    t1 = 0.01
    # Inner pipe height [m]
    h1 = 8.5
    # Outer pipe inside radius [m]
    vol1 = π * r1^2 * h1
    r2 = sqrt((vol1 + π * (r1 + t1)^2 * h1) / (h1 * π)) # <= vol1 = vol2 = π*r2^2*h1-π*(r1+t1)^2*h1
    # Outer pipe thickness [m]
    t2 = 0.01
    # Outer pipe height [m]
    h2 = 9
    # Porosity: ratio of liquid volume to the total volume
    ε = 1
    # Pipe density [g/m3]
    ρp = 961000
    # Pipe specific heat [J/(g °C)]
    cp = 2.9
    # Pipe thermal conductivity [W/(m °C)]
    λp = 0.54
    # Pipe diffusion coefficient
    dp = λp / (ρp * cp)

    # Fluid: water #########################################
    # Fluid density [g/m3]
    ρf = 997000
    # Fluid specific heat capacity [J/(g °C)]
    cf = 4.184
    # Fluid thermal conductivity [W/(m °C)]
    λf = 0.6
    # Fluid diffusion coefficient
    df = λf / (ρf * cf)
    # Flow speed [m/s]
    uf = 0.01
    vx0 = uf
    vy0 = 0
    vz0 = 0
    # Characteristic linear dimension (diameter of the pipe) [m]
    Lf = 2r1
    # Fluid dynamic viscosity at 25 °C [Pa⋅s]
    μf = 0.00089 # 0.0005465 at 50 °C
    # Reynolds number
    Re = ρf * uf * Lf / μf


    # bore hole center
    xc = 0.5
    yc = 0.5

    return (; ϕs, ρr, cr, λr, dr, ϕ0, r1, t1, h1, r2, t2, h2, ε, xc, yc,
        ρp, cp, λp, dp, ρf, cf, λf, df, uf, vx0, vy0, vz0, Lf, μf, Re)
end



function initial_condition2(x, y, z, xc, yc, all_physical_parameters)
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



r1 = 0.2
t1 = 0.05
# πr1^2 = πr2^2 - π(r1+t1)^2  =>
r2 = sqrt(2r1^2 + 2r1 * t1 + t1^2)
rp2 = r2 - r1 - t1
v = 0.01

gridz = range(-1.1 * rp2, 0.0, length=100)
gridx = range(-1.1 * r2, 0.0, length=100)

x = gridx[50]
z = gridz[50]



# uses globals
function vz_field(x, z)
    xshift = x + (r1 + 0.5 * t1)
    R = sqrt(xshift^2 + z^2)
    if R < 0.5 * t1
        vz = 0.0
    elseif R < rp2
        θ = atan(z, xshift)
        vz = v * cos(θ)
    else
        vz = 0.0
    end

    return vz
end

function vx_field(x, z)
    xshift = x + (r1 + 0.5 * t1)
    R = sqrt(xshift^2 + z^2)
    if R < 0.5 * t1
        vx = 0.0
    elseif R < rp2
        θ = atan(z, xshift)
        vx = v * sin(θ)
    else
        vx = 0.0
    end

    return vx
end


VZ = [vz_field(x, z) for x in gridx, z in gridz]
VX = [vx_field(x, z) for x in gridx, z in gridz]

heatmap(gridx, gridz, VZ', xlabel="x [m]", ylabel="z [m]", title="Velocity field vz", colorbar_title="vz [m/s]", size=(600, 500), grid=true)




using Plots

begin
    # Create parameter ranges
    θ = range(pi / 4, pi, length=50)
    φ = range(0, 2π, length=50)

    # For a torus with major radius R and minor radius r:
    R, r = 3, 1
    x = [cos(v) * (R + r * cos(u)) for u in θ, v in φ]
    y = [sin(v) * (R + r * cos(u)) for u in θ, v in φ]
    z = [-r * sin(u) for u in θ, v in φ]

    surface(x, y, z, zlims=(-1.0, 0.0))
end

φ_fixed = π / 4
x_curve = [cos(u) * (R + r * cos(φ_fixed)) for u in θ]
y_curve = [sin(u) * (R + r * cos(φ_fixed)) for u in θ]
z_curve = [r * sin(φ_fixed) for u in θ]

plot3d(x_curve, y_curve, z_curve)

# Create parameter ranges
θ = range(0, 2π, length=50)
r = range(0.1, 2, length=50)  # r now varies instead of being constant

# Fix φ to a specific value
φ_fixed = π / 4  # or whatever angle you want
R = 3

# Parametric equations with varying r and fixed φ
x = [cos(u) * (R + v * cos(φ_fixed)) for u in θ, v in r]
y = [sin(u) * (R + v * cos(φ_fixed)) for u in θ, v in r]
z = [v * sin(u) for u in θ, v in r]

plot3d(x, y, z)



# Plot several curves for different r values
θ = range(0, 2π, length=50)
r_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
φ_fixed = π / 4  # or whatever angle you want
plot3d()  # Initialize empty plot

for r_val in r_values
    x = [cos(u) * (R + r_val * cos(φ_fixed)) for u in θ]
    y = [sin(u) * (R + r_val * cos(φ_fixed)) for u in θ]
    z = [r_val * sin(φ_fixed) for u in θ]
    plot3d!(x, y, z)  # Add to existing plot
end
current()


r1 = 0.1
# Inner pipe thickness [m]
t1 = 0.01
# Inner pipe height [m]
h1 = 8.5
# Outer pipe inside radius [m]
vol1 = π * r1^2 * h1
r2full = sqrt((vol1 + π * (r1 + t1)^2 * h1) / (h1 * π)) # <= vol1 = vol2 = π*r2^2*h1-π*(r1+t1)^2*h1
r2 = r2full - r1 - t1
# Outer pipe thickness [m]
t2 = 0.01

y = 0.0
x = 0.12
x = 0.05
z = -0.01
# given x, y, z from center of torus

phi = sign(y) * acos(x / sqrt(x^2 + y^2))


r̃ = norm([x, y, z] .- [(r1 + 0.5 * t1) * cos(phi), (r1 + 0.5 * t1) * sin(phi), 0])

theta = pi - asin(abs(z) / r̃)



plot(asin)



vx = (cos(phi) * (2 * r0 * (r1 - r2) * cos(theta) + (-2 * pi * r0 * r2 + pi * (-1 + r0) * t + 2 * r0 * (-r1 + r2) * theta) * sin(theta))) / (2 * pi)

vy = (sin(phi) * (2 * r0 * (r1 - r2) * cos(theta) + (-2 * pi * r0 * r2 + pi * (-1 + r0) * t + 2 * r0 * (-r1 + r2) * theta) * sin(theta))) / (2 * pi)

vz = ((-2 * pi * r0 * r2 + pi * (-1 + r0) * t + 2 * r0 * (-r1 + r2) * theta) * cos(theta) + 2 * r0 * (-r1 + r2) * sin(theta)) / (2 * pi)




R = r1 + t1 / 2
phi = pi
n = 2
f(theta, r0) = t1 / 2 + r0 * ((1 - (theta / pi)^n) * (r2 - t1 / 2) + (theta / pi)^n * (r1 - t1 / 2))
X(theta, r0) = R * cos(phi) + f(theta, r0) * cos(theta) * cos(phi)
Y(theta, r0) = R * sin(phi) + f(theta, r0) * cos(theta) * sin(phi)
Z(theta, r0) = f(theta, r0) * -1 * sin(theta)

THETA = range(0, pi, length=100)
R0 = range(0, 1, length=8)

Xvals = [X(theta, r0) for theta in THETA, r0 in R0]
Zvals = [Z(theta, r0) for theta in THETA, r0 in R0]

function arrows(theta, r0)
    # n = 1
    vx = (cos(phi) * (2 * r0 * (r1 - r2) * cos(theta) + (-2 * pi * r0 * r2 + pi * (-1 + r0) * t1 + 2 * r0 * (-r1 + r2) * theta) * sin(theta))) / (2 * pi)

    vy = (sin(phi) * (2 * r0 * (r1 - r2) * cos(theta) + (-2 * pi * r0 * r2 + pi * (-1 + r0) * t1 + 2 * r0 * (-r1 + r2) * theta) * sin(theta))) / (2 * pi)

    vz = ((-2 * pi * r0 * r2 + pi * (-1 + r0) * t1 + 2 * r0 * (-r1 + r2) * theta) * cos(theta) + 2 * r0 * (-r1 + r2) * sin(theta)) / (2 * pi)

    # n = 2
    vx = -0.5 * (cos(phi) * (4 * r0 * (-r1 + r2) * theta * cos(theta) +
                             (pi^2 * (2 * r0 * r2 + t1 - r0 * t1) + 2 * r0 * (r1 - r2) * theta^2) * sin(theta))) / pi^2

    vy = -0.5 * (sin(phi) * (4 * r0 * (-r1 + r2) * theta * cos(theta) +
                             (pi^2 * (2 * r0 * r2 + t1 - r0 * t1) + 2 * r0 * (r1 - r2) * theta^2) * sin(theta))) / pi^2

    vz = ((-(pi^2 * (2 * r0 * r2 + t1 - r0 * t1)) + 2 * r0 * (-r1 + r2) * theta^2) * cos(theta) + 4 * r0 * (-r1 + r2) * theta * sin(theta)) / (2 * pi^2)


    return 0.01 * vx, 0.01 * vz
end

# scatter(Xvals, Zvals)
# scatter!(Xvals[:,7], Zvals[:,7])





# Subsample your parameter space to avoid overcrowding
theta_vec = range(0, pi, length=50)  # fewer points than your surface
r0_vec = range(0, 1, length=8)

# Compute positions and vector components at each point
X_arrows = []
Z_arrows = []
VX_arrows = []
VZ_arrows = []

for theta in theta_vec, r0 in r0_vec
    # Position
    x_pos = X(theta, r0)
    z_pos = Z(theta, r0)

    # Vector components
    vx, vz = arrows(theta, r0)

    push!(X_arrows, x_pos)
    push!(Z_arrows, z_pos)
    push!(VX_arrows, vx)
    push!(VZ_arrows, vz)
end

# Plot
scatter(Xvals, Zvals, alpha=0.6, label="Surface points")
quiver!(X_arrows, Z_arrows, quiver=(VX_arrows, VZ_arrows),
    color=:red, alpha=0.8, label="Vector field")







# check inverse equations
X1(theta, phi, r0) = R * cos(phi) + f(theta, r0) * cos(theta) * cos(phi)
Y1(theta, phi, r0) = R * sin(phi) + f(theta, r0) * cos(theta) * sin(phi)
Z1(theta, phi, r0) = f(theta, r0) * -1 * sin(theta)

begin
    theta = rand() * pi
    phi = 2(rand() - 0.5)pi
    r0 = rand()
    x = X1(theta, phi, r0)
    y = Y1(theta, phi, r0)
    z = Z1(theta, phi, r0)

    phi_ = atan(y, x)

    rho = sqrt(x^2 + y^2)
    r̃ = sqrt((rho - (r1 + 0.5 * t1))^2 + z^2)
    sin_theta = -z / r̃
    cos_theta = (rho - R) / r̃
    theta_ = atan(sin_theta, cos_theta)    # This handles all quadrants correctly

    r0_ = (r̃ - t1 / 2) / ((1 - (theta_ / pi)^2) *
                           (r2 - t1 / 2) + (theta_ / pi)^2 * (r1 - t1 / 2))

    x2 = X1(theta_, phi_, r0_)
    y2 = Y1(theta_, phi_, r0_)
    z2 = Z1(theta_, phi_, r0_)

    x ≈ x2, y ≈ y2, z ≈ z2
end

theta = 0
vx = -0.5 * (cos(phi) * (4 * r0 * (-r1 + r2) * theta * cos(theta) +
                         (pi^2 * (2 * r0 * r2 + t1 - r0 * t1) + 2 * r0 * (r1 - r2) * theta^2) * sin(theta))) / pi^2

vy = -0.5 * (sin(phi) * (4 * r0 * (-r1 + r2) * theta * cos(theta) +
                         (pi^2 * (2 * r0 * r2 + t1 - r0 * t1) + 2 * r0 * (r1 - r2) * theta^2) * sin(theta))) / pi^2

vz = ((-(pi^2 * (2 * r0 * r2 + t1 - r0 * t1)) + 2 * r0 * (-r1 + r2) * theta^2) * cos(theta) + 4 * r0 * (-r1 + r2) * theta * sin(theta)) / (2 * pi^2)

