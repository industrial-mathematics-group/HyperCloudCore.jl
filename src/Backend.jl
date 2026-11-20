# =========================================================================
# EXECUTION BACKENDS
#
# Every parallel per-particle loop of the package is written as a *kernel body*
# `body(i, args...)` that only touches the arrays and structs it is given, and is launched with
# `foreach_particle(body, backend, N, args)`. The backend decides how the loop runs:
#
#   PolyesterBackend()   Polyester `@batch per=core`        (default; fastest on the CPU; one worker per
#                        physical core: spin-waiting workers collapse when they oversubscribe SMT siblings)
#   ThreadsBackend()     Base.Threads `@threads :dynamic`    (no spin-waiting workers)
#   KABackend(b)         a KernelAbstractions.jl kernel on `b` (`CPU()`, or a GPU backend such as
#                        `CUDABackend()` / `ROCBackend()` from the vendor package)
#
# Bodies therefore have to obey the GPU kernel rules: no allocation, no dynamic dispatch, no
# `Dict` access, nothing captured from the enclosing method's static parameters. Two Polyester rules
# apply to every `@batch` loop of the package: the body must not use the static parameters of the
# enclosing method (`T`, `D`, `Val(D)`, `zero(T)`), Polyester passes them into the loop closure as
# runtime objects and evaluates such expressions dynamically per iteration; and every captured
# variable must be assigned exactly once in the enclosing function, a reassigned captured variable is
# boxed and every access allocates. Serial parts (prefix sums, the MOOD decrement, sorting, boundary
# donor searches) stay on the host for now; see docs/src/gpu_roadmap.md.
# =========================================================================
using KernelAbstractions
using Adapt
const KA = KernelAbstractions

export ExecutionBackend, PolyesterBackend, ThreadsBackend, KABackend
export set_backend!, get_backend, foreach_particle

"Supertype of the ways a per-particle loop can be executed (see `Backend.jl`)."
abstract type ExecutionBackend end

"Polyester `@batch per=core` loops: the default CPU path."
struct PolyesterBackend <: ExecutionBackend end

"`Threads.@threads :dynamic` loops (no spin-waiting workers; handy next to other threaded code)."
struct ThreadsBackend <: ExecutionBackend end

"""
    KABackend(backend::KernelAbstractions.Backend; workgroupsize = 256)

Runs every per-particle loop as a KernelAbstractions kernel on `backend`: `CPU()` for testing the
kernels on the host, or a device backend from CUDA.jl / AMDGPU.jl / Metal.jl / oneAPI.jl. All
arrays reached by the kernels must live on that device (see docs/src/gpu_roadmap.md).

Kernels are launched with work-groups of `workgroupsize` items. A kernel whose own limit is smaller
(its private memory or registers grow with the state size: on PoCL the 2-D flux kernel of a
36-component state allows 231 items) is launched with a work-group size KernelAbstractions chooses
within that limit instead; the launch is checked before anything runs, and the choice is kept for
that kernel (body and argument types) from then on.
"""
struct KABackend{B <: KA.Backend} <: ExecutionBackend
    backend::B
    workgroupsize::Int
end
KABackend(backend::KA.Backend; workgroupsize::Int = 256) = KABackend(backend, workgroupsize)

const _BACKEND = Ref{ExecutionBackend}(PolyesterBackend())

"The execution backend of the per-particle loops (default `PolyesterBackend()`)."
get_backend() = _BACKEND[]

"""
    set_backend!(backend)

Selects how the per-particle loops run: an `ExecutionBackend`, or a `KernelAbstractions.Backend`
(wrapped in a `KABackend`). Returns the new backend.
"""
set_backend!(backend::ExecutionBackend) = (_BACKEND[] = backend)
set_backend!(backend::KA.Backend; kwargs...) = set_backend!(KABackend(backend; kwargs...))

# The backend to launch the kernels of a grid's own arrays with: the session backend, unless that is a device
# backend and the grid lives on the host (`device_grid(Array, pg)`: the sweep of moving boundaries searches the
# neighbours of its host copy between its rounds, which was launched on the device and failed on the host arrays)
_launch_backend(pg) = _launch_backend(get_backend(), parent(pg.core.positions))
_launch_backend(b::ExecutionBackend, ::AbstractArray) = b
_launch_backend(b::KABackend, x::AbstractArray) = KA.get_backend(x) == b.backend ? b : PolyesterBackend()

"""
    KernelView

`Adapt` target that turns host objects into the form a kernel body sees (see `DeviceAdapt.jl`): mutable
metadata becomes an immutable snapshot, `Ref`s plain values, `Growable` buffers the arrays they hold, and
host-only parts are dropped. Arrays are left where they are. Every `foreach_particle` launch adapts its
arguments this way, so kernel bodies see the same argument types on every backend.
"""
struct KernelView end

"The kernel-body form of `x` (see `KernelView`)."
kernel_view(x) = adapt(KernelView(), x)

"""
    foreach_particle(body, backend, N, args::Tuple)

Runs `body(i, args...)` for `i in 1:N` on `backend`. `body` must be a kernel body in the sense of
`Backend.jl` (no allocation, no dynamic dispatch, only `args` as input). The arguments are passed
through `kernel_view` first; on a `KABackend` every array among them must live on that backend.
"""
function foreach_particle(body::F, ::PolyesterBackend, N::Int, args::Tuple) where {F}
    return _polyester_loop(body, N, kernel_view(args))
end

# Function barrier: with a single Julia thread `@batch` runs the loop inline instead of through its closure,
# so an argument tuple whose type is not inferred here would make every `body` call a dynamic dispatch
# that boxes its arguments (per particle). The barrier specialises the loop on the runtime types, as the
# closures of the threaded paths do.
function _polyester_loop(body::F, N::Int, kargs::A) where {F, A}
    @batch per=core for i in 1:N
        body(i, kargs...)
    end
    return nothing
end

function foreach_particle(body::F, ::ThreadsBackend, N::Int, args::Tuple) where {F}
    kargs = kernel_view(args)
    Threads.@threads :dynamic for i in 1:N
        body(i, kargs...)
    end
    return nothing
end

@kernel function _foreach_kernel(body, args)
    i = @index(Global, Linear)
    body(i, args...)
end

function foreach_particle(body::F, b::KABackend, N::Int, args::Tuple) where {F}
    N == 0 && return nothing
    kargs = kernel_view(args)
    check_resident(b.backend, kargs)
    key = (b.backend, b.workgroupsize, typeof(body), typeof(kargs))
    if _needs_tuned_workgroup(key)
        _foreach_kernel(b.backend)(body, kargs; ndrange = N)
    else
        try
            _foreach_kernel(b.backend, b.workgroupsize)(body, kargs; ndrange = N)
        catch err
            _is_workgroup_size_error(err) || rethrow()
            # The kernel allows fewer work-items per group than `b.workgroupsize`. The launch was checked
            # before anything ran, so relaunch with a size KernelAbstractions picks within the kernel's limit.
            _remember_tuned_workgroup!(key)
            _foreach_kernel(b.backend)(body, kargs; ndrange = N)
        end
    end
    KA.synchronize(b.backend)
    return nothing
end

# =========================================================================
# LAUNCHES OVER (PARTICLE, SLOT) PAIRS
#
# `foreach_pairs(body, backend, N, nc, args)` runs `body(i, c, args...)` for `i in 1:N`, `c in 1:nc`. On a device it is a
# two-dimensional launch: every work-item gets its particle and its slot from the launch itself. The launch over
# `N · nc` linear work-items (what `foreach_block` did) computed `(q − 1) % N` and `(q − 1) ÷ N` in every work-item, a
# 64-bit integer division that GPUs emulate in software (no divide instruction). Measured on a V100 the saving is small
# (0–1 % of a block step: the light passes were slow for another reason, their stage tuples in local memory), but the
# launch is no more complex. The CPU backends run the linear loop (a division is cheap there).
# =========================================================================
@kernel function _pairs_kernel(body, args)
    i, c = @index(Global, NTuple)
    body(i, c, args...)
end
# The slots along the fast dimension: the work-items of a warp are consecutive slots (blocks) of the same particle, so the
# particle's data (its neighbour list, distances, weights, fit) is one broadcast load per warp instead of one load per
# work-item, and the lanes of a warp loop over the same neighbours (with the particles along the warp they diverged)
@kernel function _pairs_kernel_slots(body, args)
    c, i = @index(Global, NTuple)
    body(i, c, args...)
end
# Which of the two layouts the device launches use (an experiment switch; `true`: slots along the warp)
const _SLOTS_FASTEST = Ref(false)
# (the grid's second dimension holds the slots: CUDA allows 65535 there; more slots run the linear launch)
const _MAX_PAIR_SLOTS = 65535

function foreach_pairs(body::F, backend::ExecutionBackend, N::Int, nc::Int, args::Tuple) where {F}
    (N == 0 || nc == 0) && return nothing
    return foreach_particle(_pairs_linear_body!, backend, N * nc, (N, body, args))
end
@inline function _pairs_linear_body!(q, N, body, args)
    body((q - 1) % N + 1, (q - 1) ÷ N + 1, args...)
    return nothing
end
function foreach_pairs(body::F, b::KABackend, N::Int, nc::Int, args::Tuple) where {F}
    (N == 0 || nc == 0) && return nothing
    nc > _MAX_PAIR_SLOTS && return foreach_particle(_pairs_linear_body!, b, N * nc, (N, body, args))
    kargs = kernel_view(args)
    check_resident(b.backend, kargs)
    key = (b.backend, b.workgroupsize, typeof(body), typeof(kargs), :pairs)
    if _SLOTS_FASTEST[] && nc > 1
        wx = min(nc, 32)
        wy = max(1, b.workgroupsize ÷ wx)
        if _needs_tuned_workgroup(key)
            _pairs_kernel_slots(b.backend)(body, kargs; ndrange = (nc, N))
        else
            try
                _pairs_kernel_slots(b.backend, (wx, wy))(body, kargs; ndrange = (nc, N))
            catch err
                _is_workgroup_size_error(err) || rethrow()
                _remember_tuned_workgroup!(key)
                _pairs_kernel_slots(b.backend)(body, kargs; ndrange = (nc, N))
            end
        end
    elseif _needs_tuned_workgroup(key)
        _pairs_kernel(b.backend)(body, kargs; ndrange = (N, nc))
    else
        try
            _pairs_kernel(b.backend, (b.workgroupsize, 1))(body, kargs; ndrange = (N, nc))
        catch err
            _is_workgroup_size_error(err) || rethrow()
            _remember_tuned_workgroup!(key)                 # (as in `foreach_particle`)
            _pairs_kernel(b.backend)(body, kargs; ndrange = (N, nc))
        end
    end
    KA.synchronize(b.backend)
    return nothing
end

# Kernels (backend, configured size, body and argument types) whose limit is below the configured work-group size
const _TUNED_WORKGROUP = Set{Any}()
const _TUNED_WORKGROUP_LOCK = ReentrantLock()
_needs_tuned_workgroup(key) = !isempty(_TUNED_WORKGROUP) && lock(() -> key in _TUNED_WORKGROUP, _TUNED_WORKGROUP_LOCK)
_remember_tuned_workgroup!(key) = lock(() -> push!(_TUNED_WORKGROUP, key), _TUNED_WORKGROUP_LOCK)
# KernelInterface's check of a launch against the kernel's limits (thrown before the launch)
_is_workgroup_size_error(err) = err isa ArgumentError && occursin("Work-group size", err.msg)

"""
    check_resident(backend::KernelAbstractions.Backend, x)

Throws an `ArgumentError` if any array reachable from `x` (through tuples and struct fields) does not live
on `backend`. Guards the kernel launches: AMDGPU.jl, for one, would otherwise copy a host array to the
device for the launch and silently drop the kernel's writes to it.
"""
@generated function check_resident(backend, x)
    (isbitstype(x) || x <: Type || !isconcretetype(x)) && return :(nothing)
    checks = [:(check_resident(backend, getfield(x, $i))) for i in 1:fieldcount(x)]
    return quote
        $(checks...)
        return nothing
    end
end

function check_resident(backend, x::AbstractArray)
    isbits(x) && return nothing
    KA.get_backend(x) == backend && return nothing
    throw(ArgumentError("a kernel argument of type $(typeof(x)) is not resident on $(backend); build the grid " *
                        "with `device_grid` and the interpolator and time stepper with the matching `storage` type"))
end

# =========================================================================
# LAUNCHES OVER (PARTICLE, BLOCK) PAIRS
# =========================================================================
"""
    foreach_block(body, backend, N, blocks::UnitRange, args::Tuple)

Runs the kernel body `body(i, args_b...)` for every particle `i in 1:N` and every block `b in blocks` in **one
launch** of `N · length(blocks)` work-items: the passes of the block pipeline over a block-major state, instead
of one launch of `N` work-items per block (which never fills a GPU and pays the launch latency once per block).
The work-items are the pairs (particle `i`, `c`-th block of the launch), `b = first(blocks) + c − 1` (`foreach_pairs`:
a two-dimensional launch on a device): consecutive work-items are consecutive particles of one block, the order of the
block-major layout. The
arguments are those of the per-particle launch, with the per-block ones marked:

- `PerBlock``(a)`: a block-major array (`nb N` elements, block `b` at `(b − 1) N + 1 … b N`); the body
  gets the view of block `b`;
- `PerBlockEq``(eq)`: the body gets `BlockOf(eq, b + block_offset)`;
- `PerBlockIndex``()`: the body gets `b + block_offset`;
- `PerSlot``(x, stride)`: scratch of one launch slot (an `InteractionBuffer`, an interpolator with
  per-particle caches): the body gets `x` with its arrays shifted by `(c − 1) stride` (`_slot_view`), so the
  blocks of one launch write disjoint scratch; the caller sizes `x` for `length(blocks)` slots;
- `PerSlotEntry``(a)`: the body gets `a[c]`;
- a tuple: its elements are translated one by one; anything else is passed through unchanged.

`block_offset`: the arrays hold the blocks `blocks` of a block range starting at `block_offset + 1` (a part of a
state split over devices): the views are those of the local block `b`, the equation and the index get the global
block `b + block_offset`.

`list = (indices, n)`: only the particles `indices[1:n]` (see `foreach_listed`), `n · length(blocks)`
work-items.
"""
function foreach_block(body::F, backend::ExecutionBackend, N::Int, blocks::UnitRange{Int}, args::Tuple; block_offset::Int = 0,
                       list = nothing) where {F}
    (N == 0 || isempty(blocks)) && return nothing
    if list !== nothing                               # only the listed particles (`list = (indices, n)`)
        idx, n = list
        return foreach_pairs(_listed_pairs_body!, backend, n, length(blocks), (idx, N, first(blocks), block_offset, body, args))
    end
    return foreach_pairs(_block_pairs_body!, backend, N, length(blocks), (N, first(blocks), block_offset, body, args))
end

"""
    foreach_listed(body, backend, indices, n, args::Tuple)

Runs `body(i, args...)` for the particles `i = indices[k]`, `k in 1:n`, in one launch of `n` work-items: a pass that
only some particles do (the ghosts of one boundary tag), without the launch over all `N` particles in which most
work-items return at once and a warp with one listed particle costs as much as a full one. `indices` lives on the
backend's storage (a host `Vector` for the CPU backends). `foreach_block(…; list = (indices, n))` is the launch over
the (listed particle, block) pairs.
"""
function foreach_listed(body::F, backend::ExecutionBackend, indices, n::Int, args::Tuple) where {F}
    n == 0 && return nothing
    return foreach_particle(_listed_body!, backend, n, (indices, body, args))
end
@inline function _listed_body!(k, indices, body, args)
    @inbounds i = indices[k]
    body(i, args...)
    return nothing
end
# The pair (listed particle `k`, `c`-th block of the launch) of a launch over the listed particles and the blocks (the
# views of the blocks have `N` entries, as in `_block_pairs_body!`)
@inline function _listed_pairs_body!(k, c, indices, N, b0, off, body, args)
    b = b0 + c - 1
    @inbounds i = indices[k]
    body(i, map(x -> _block_arg(x, b, c, N, off), args)...)
    return nothing
end

"A block-major array argument of `foreach_block`: the body gets the view of the block."
struct PerBlock{A}
    a::A
end
"An equation argument of `foreach_block`: the body gets `BlockOf(eq, b)`, or `BlockOf(eq, b, table[b])` with a table of the blocks' constants."
struct PerBlockEq{EQ, B}
    eq::EQ
    table::B
end
PerBlockEq(eq) = PerBlockEq(eq, nothing)
"The block index as an argument of `foreach_block`."
struct PerBlockIndex end
"Scratch of one launch slot of `foreach_block`: the body gets `x` shifted by `(c − 1) stride`."
struct PerSlot{X}
    x::X
    stride::Int
end
"An array indexed by the launch slot in `foreach_block`: the body gets `a[c]`."
struct PerSlotEntry{A}
    a::A
end
Adapt.adapt_structure(to, x::PerBlock) = PerBlock(adapt(to, x.a))
Adapt.adapt_structure(to, x::PerBlockEq) = PerBlockEq(adapt(to, x.eq), adapt(to, x.table))
Adapt.adapt_structure(to, x::PerSlot) = PerSlot(adapt(to, x.x), x.stride)
Adapt.adapt_structure(to, x::PerSlotEntry) = PerSlotEntry(adapt(to, x.a))

# The argument block `b` (launch slot `c`) sees; `_block_arg(::PerBlockEq)` and the `_slot_view`s of the scratch
# objects are defined with them (`BlockStepper.jl`)
@inline _block_arg(x, b, c, N, off) = x
@inline _block_arg(x::PerBlock, b, c, N, off) = @inbounds view(x.a, ((b - 1) * N + 1):(b * N))
@inline _block_arg(x::Tuple, b, c, N, off) = map(y -> _block_arg(y, b, c, N, off), x)
@inline _block_arg(::PerBlockIndex, b, c, N, off) = b + off
@inline _block_arg(x::PerSlot, b, c, N, off) = _slot_view(x.x, (c - 1) * x.stride, x.stride)
@inline _block_arg(x::PerSlotEntry, b, c, N, off) = @inbounds x.a[c]
@inline _slot_view(x, off::Int, len::Int) = x

# Kernel body (see above): the pair (particle `i`, `c`-th block of the launch), `b` the local block, `b + off` the global one
@inline function _block_pairs_body!(i, c, N, b0, off, body, args)
    b = b0 + c - 1
    body(i, map(x -> _block_arg(x, b, c, N, off), args)...)
    return nothing
end
