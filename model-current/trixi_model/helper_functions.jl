
function set_up_physics(Lx, Ly, Lz)
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

    return (; Lx, Ly, Lz, ϕs, ρr, cr, λr, dr, ϕ0, r1, t1, h1, r2, t2, h2, ε, xc, yc,
        ρp, cp, λp, dp, ρf, cf, λf, df, uf, vx0, vy0, vz0, Lf, μf, Re)
end


