# Cuboid signed distance for points outside by less than sqrt(floatmin): the square underflows.
using HyperCloudCore, StaticArrays
g = get_rectangular_domain(Float64, (0.0, 0.0), (1.0, 1.0))
x = SVector(-1e-170, 0.5)
println("F64 x = $x: is_interior ", g.is_interior(x), ", sd ", signed_distance(g.shape, x), ", tag ", g.get_tag(x))
g32 = get_rectangular_domain(Float32, (0f0,), (1f0,))
y = SVector(1f0 + 0f0, ); z = SVector(-1f-25)
println("F32 x = $z: is_interior ", g32.is_interior(z), ", tag ", g32.get_tag(z))
