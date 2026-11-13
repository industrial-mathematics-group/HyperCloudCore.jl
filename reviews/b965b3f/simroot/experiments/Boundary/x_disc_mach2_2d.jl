# X5 (review): impulsively started Mach 2 flow past a static disc (bow shock on a curved slip wall), t = 0.4: stability
# and positivity only (FixedDirichlet far field).
isdefined(@__MODULE__, :BND_METHODS) || Base.include(@__MODULE__, joinpath(@__DIR__, "boundary_common.jl"))
function get_config()
    d = FixedDirichlet()
    shared = bnd_shared(2, :tmax => 0.4, :Grid_domain => :shape, :Grid_shape => "Cuboid((T(-1), T(-1)), (T(1), T(1))) - Tagged(Ball((T(0), T(0)), T(0.15)), 5)",
                        :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (200, 200),
                        :Grid_bc => BND_BCD(1 => d, 2 => d, 3 => d, 4 => d, 5 => slip_wall()),
                        bnd_uniform(2, bnd_cons(1.0, (2 * sqrt(1.4), 0.0), 1.0))..., :disc_radius => 0.15)
    varied = Dict{Symbol, Vector}(:Grid_Ns => [(200, 200)])
    return bnd_config(shared, BND_METHODS_MD; varied = varied, sources = [@__FILE__])
end
