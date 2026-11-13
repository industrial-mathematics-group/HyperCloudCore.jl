# X1 (review): a disc translating at (0.5, 0) through a gas moving with it (ρ = p = 1, u = (0.5, 0)): uniform state is exact.
isdefined(@__MODULE__, :BND_METHODS) || Base.include(@__MODULE__, joinpath(@__DIR__, "boundary_common.jl"))
function get_config()
    d = FixedDirichlet()
    shared = bnd_shared(2, :tmax => 0.3, :Grid_domain => :shape, :Grid_shape => "Cuboid((T(-1), T(-1)), (T(1), T(1))) - Moving(Tagged(Ball((T(-0.075), T(0)), T(0.15)), 5), translating((T(0.5), T(0))))",
                        :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (100, 100),
                        :Grid_bc => BND_BCD(1 => d, 2 => d, 3 => d, 4 => d, 5 => slip_wall()),
                        bnd_uniform(2, bnd_cons(1.0, (0.5, 0.0), 1.0))..., :disc_radius => 0.15)
    varied = Dict{Symbol, Vector}(:Grid_Ns => [(100, 100), (200, 200)])
    return bnd_config(shared, BND_METHODS_MD; varied = varied, sources = [@__FILE__])
end
