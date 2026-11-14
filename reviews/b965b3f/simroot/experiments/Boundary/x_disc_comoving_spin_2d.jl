# X4 (review): the disc of X1 translating with the gas and spinning about its own centre (ω = 2): a slip wall ignores
# the tangential wall velocity, so the uniform state is exact.
isdefined(@__MODULE__, :BND_METHODS) || Base.include(@__MODULE__, joinpath(@__DIR__, "boundary_common.jl"))
function get_config()
    d = FixedDirichlet()
    mot = "t -> (c0 = SVector{2, T}(-0.075, 0); V = SVector{2, T}(0.5, 0); J = SMatrix{2, 2, T}(0, 1, -1, 0); R = SMatrix{2, 2, T}(cos(2t), sin(2t), -sin(2t), cos(2t)); RigidMotion(c0 + V * T(t) - R * c0, R, V - 2J * R * c0, 2J))"
    shared = bnd_shared(2, :tmax => 0.3, :Grid_domain => :shape, :Grid_shape => "Cuboid((T(-1), T(-1)), (T(1), T(1))) - Moving(Tagged(Ball((T(-0.075), T(0)), T(0.15)), 5), $mot)",
                        :Grid_mins => (-1.0, -1.0), :Grid_maxs => (1.0, 1.0), :Grid_Ns => (100, 100),
                        :Grid_bc => BND_BCD(1 => d, 2 => d, 3 => d, 4 => d, 5 => slip_wall()),
                        bnd_uniform(2, bnd_cons(1.0, (0.5, 0.0), 1.0))..., :disc_radius => 0.15)
    varied = Dict{Symbol, Vector}(:Grid_Ns => [(100, 100), (200, 200)])
    return bnd_config(shared, BND_METHODS_MD; varied = varied, sources = [@__FILE__])
end
