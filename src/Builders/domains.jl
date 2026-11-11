# =========================================================================
# MODULAR DOMAIN BUILDER
# =========================================================================

# --- Boundary Condition Parsing ---
"""
    parse_bc(domain_conf) -> Dict{Int, AbstractBoundaryCondition}

The `bc_map` of a domain from `domain_conf[:bc]` (or `:bc_map`): a dictionary `tag => bc`, where `bc` is an
[`AbstractBoundaryCondition`](@ref) instance, `:outflow`/`:OutflowBC`, `:fixed_dirichlet`/`:FixedDirichlet`,
or the name of a zero-argument constructor defined in HyperCloudCore or in `Main`. A string is parsed and
evaluated as Julia code first. Without the key the map is empty.
"""
function parse_bc end

function parse_bc(domain_conf::Dict)
    raw_bc = get(domain_conf, :bc, get(domain_conf, :bc_map, Dict{Int, AbstractBoundaryCondition}()))
    
    if raw_bc isa String
        raw_bc = eval(Meta.parse(raw_bc))
    end
    
    bc_map = Dict{Int, AbstractBoundaryCondition}()
    
    for (tag, bc_obj) in raw_bc
        if bc_obj isa AbstractBoundaryCondition
            bc_map[tag] = bc_obj
        else
            bc_sym = Symbol(bc_obj)
            if bc_sym === :outflow || bc_sym === :OutflowBC
                bc_map[tag] = OutflowBC()
            elseif bc_sym === :fixed_dirichlet || bc_sym === :FixedDirichlet
                bc_map[tag] = FixedDirichlet()
            else
                if isdefined(HyperCloudCore, bc_sym)
                    bc_map[tag] = getfield(HyperCloudCore, bc_sym)()
                elseif isdefined(Main, bc_sym)
                    bc_map[tag] = getfield(Main, bc_sym)()
                else
                    error("Boundary Condition '$bc_sym' could not be found.")
                end
            end
        end
    end
    
    return bc_map
end

# --- Main Domain Builder ---
"""
    build_domain(domain_conf, context) -> GeometricDomain
    build_domain(::Val{name}, domain_conf, context)

`Grid` namespace. Selector `:domain` (a `Symbol` or `String`):

- `:rectangular`: [`get_rectangular_domain`](@ref) on the box `:mins`, `:maxs`;
- `:spherical`: [`get_spherical_domain`](@ref) with the midpoint of the box as centre and half its first
  extent as radius (needs `context[:D]`);
- `:shape`: the [`AbstractShape`](@ref) in `:shape` (an instance, or a string of Julia code that evaluates to
  one, in which `T` is the run's float type: `"Ball((T(0), T(0)), T(0.5))"` works for Float32 and Float64), with
  `:mins`/`:maxs` where the shape is unbounded (optional otherwise). A string is also what the parameter hash
  sees in full: the motion of a `Moving` piece is a function, which the hash of an instance does not see.

All read `:periodic` (`Bool` or `NTuple{D, Bool}`, default `false`) and the boundary conditions
(`:bc`, see [`parse_bc`](@ref)). A [`GeometricDomain`](@ref) among the values of `domain_conf` is returned
as it is, so a pre-built domain can be passed through the configuration.
"""
function build_domain end

function build_domain(domain_conf::Dict, context::Dict)

    for (_,val) in domain_conf
        if val isa GeometricDomain; return val end
    end

    if !haskey(domain_conf, :domain)
        error("Domain configuration must strictly include a :domain key (e.g., :rectangular, :spherical).")
    end

    domain_name = Symbol(domain_conf[:domain])   # accepts Symbol or String
    return build_domain(Val(domain_name), domain_conf, context)
end

# Generic fallback
build_domain(name::Val, domain_conf::Dict, context::Dict) = error("Unknown domain shape: $(typeof(name))")

# --- Specific Domain Builders ---
function build_domain(::Val{:rectangular}, domain_conf::Dict, context::Dict)
    T = real_type(context)
    
    bc_map = parse_bc(domain_conf)
    
    # Directly pull periodicity (expected to be Bool or NTuple{D, Bool})
    is_per = get(domain_conf, :periodic, false)
    
    req_mins = Tuple(T.(domain_conf[:mins]))
    req_maxs = Tuple(T.(domain_conf[:maxs]))
    
    return get_rectangular_domain(T, req_mins, req_maxs; bc_map = bc_map, is_periodic = is_per)
end

function build_domain(::Val{:shape}, domain_conf::Dict, context::Dict)
    haskey(domain_conf, :shape) || error("Domain :shape needs the key :shape (an AbstractShape)")
    shape = domain_conf[:shape]
    T = real_type(context)
    # A string is evaluated in `Main`, where the user's own shape types live (HyperCloudCore's shapes are visible
    # there through `using HyperCloud`/`using HyperCloudCore`); HyperCloudCore itself if `Main` doesn't know a name.
    # `T` is bound to the run's float type.
    if shape isa String
        ex = :(let T = $T; $(Meta.parse(shape)); end)
        shape = try
            Core.eval(Main, ex)
        catch e
            e isa UndefVarError || rethrow()
            eval(ex)
        end
    end
    shape isa AbstractShape || error("Domain :shape: :shape must be an AbstractShape, got $(typeof(shape))")
    eltype(shape) == T ||
        error("Domain :shape: the shape has element type $(eltype(shape)), the run uses $T; build the shape in $T")
    haskey(context, :D) && ndims(shape) != context[:D] &&
        error("Domain :shape: the shape is $(ndims(shape))-D, the run is $(context[:D])-D")
    bc_map = parse_bc(domain_conf)
    is_per = get(domain_conf, :periodic, false)
    mins = haskey(domain_conf, :mins) ? Tuple(T.(domain_conf[:mins])) : nothing
    maxs = haskey(domain_conf, :maxs) ? Tuple(T.(domain_conf[:maxs])) : nothing
    return GeometricDomain(shape; bc_map = bc_map, is_periodic = is_per, mins = mins, maxs = maxs)
end

function build_domain(::Val{:spherical}, domain_conf::Dict, context::Dict)
    T = real_type(context)
    D = context[:D]::Int
    
    bc_map = parse_bc(domain_conf)
    
    # Directly pull periodicity
    is_per = get(domain_conf, :periodic, false)
    
    req_mins = Tuple(T.(domain_conf[:mins]))
    req_maxs = Tuple(T.(domain_conf[:maxs]))
    
    # The sphere is inscribed in the box: centre at its middle, radius half its x extent (the other extents are not
    # used, so a non-cubic box gives the sphere of its first axis).
    center = ntuple(d -> (req_mins[d] + req_maxs[d]) / 2, Val(D))
    radius = (req_maxs[1] - req_mins[1]) / 2
    
    return get_spherical_domain(T, center, radius; bc_map = bc_map, is_periodic = is_per)
end