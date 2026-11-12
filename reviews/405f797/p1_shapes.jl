using HyperCloudCore, StaticArrays, LinearAlgebra
const H = HyperCloudCore
println("== FD check of velocities (2-D, 3-D rotating, arbitrary axis)")
for (m, D) in ((rotating(1.7; center=(0.3,-0.2)), 2), (rotating((1.0,2.0,-0.5), 1.3; center=(0.1,0.2,0.3)), 3))
    t = 0.37; ε = 1e-6
    y = SVector{D}(ntuple(i->0.4+0.1i, D))
    p = m(t); x = H.body_to_world(p, y)
    xd = (H.body_to_world(m(t+ε), y) - H.body_to_world(m(t-ε), y)) / 2ε
    println("D=$D |v_fd - body_velocity| = ", norm(xd - H.body_velocity(p, x)), "  R orth err ", norm(p.R'p.R - I), " det ", det(p.R))
    # center fixed?
    c0 = SVector{D}(D==2 ? (0.3,-0.2) : (0.1,0.2,0.3))
    println("   center fixed: ", norm(H.body_to_world(p, c0) - c0))
end
println("== Float32")
for f in (() -> Moving(Ball((0f0,0f0), 0.2f0), rotating(1f0)),
          () -> Moving(Ball((0f0,0f0), 0.2f0), rotating(1.0)),
          () -> Moving(Ball((0f0,0f0), 0.2f0), rotating(1f0; center=(0f0,0f0))),
          () -> Moving(Ball((0f0,0f0,0f0), 0.2f0), rotating((0f0,0f0,1f0), 1f0)),
          () -> Moving(Ball((0f0,0f0), 0.2f0), translating((1.0, 0.0))),
          () -> Moving(Ball((0f0,0f0), 0.2f0), translating((1f0, 0f0))))
    try; m = f(); println("ok ", typeof(m.motion(0f0))); catch e; println("ERR ", sprint(showerror, e)[1:min(end,140)]); end
end
println("== Moving inside Rotated / Translated")
ball = Tagged(Ball((0.5, 0.0), 0.1), 5)
mv = Moving(ball, translating((1.0, 0.0)))
Rr = Rotated(mv, π/2)   # world: moves along +y
s = complement(Rr)
t = 0.1
x = SVector(0.0, 0.5 + 0.1 + 0.1 + 0.0)   # near the rotated ball at t=0.1: center (0, 0.5+0.1)
println("rotated shape center at t (from sd minimum): sd at (0,0.6) = ", signed_distance(at_time(s, t), SVector(0.0, 0.6)))
println("wall_velocity (expected (0,1)): ", wall_velocity(s, SVector(0.0, 0.75), t))
mv2 = Moving(Tagged(Ball((0.2, 0.0), 0.1), 5), rotating(1.0))   # rotates about local origin
Tr = Translated(mv2, (1.0, 0.0))  # world rotation centre (1,0)
s2 = complement(Tr)
# at t=pi/2 the ball centre is at world (1,0)+(0,0.2) = (1,0.2)
println("sd at (1,0.2), t=pi/2 (expect -0.1 => inside the ball, complement +0.1): ", signed_distance(at_time(s2, π/2), SVector(1.0, 0.2)))
println("wall_velocity at (1, 0.35) t=pi/2 (expected ω×(x-c)=(-0.35,0)): ", wall_velocity(s2, SVector(1.0, 0.35), π/2))
