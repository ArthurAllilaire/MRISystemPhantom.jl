# Random phantom configuration: sample complete deterministic `PhantomConfig`s
# for RL training episodes while keeping the true sampled configuration as a
# hidden `truth` record for evaluation. See RANDOM_PHANTOM_PLAN.md.
#
# The whole API is duck-typed: samplers may be `Distributions.jl` objects,
# constants, closures, or callable structs. The library never imports
# `Distributions` — it only calls `rand(rng, x)` (via `_sample`).

# Plates that the randomisation pipeline may select/sample. `:fiducials` stays
# deterministic by default and `:water` is never a sphere-descriptor plate.
const _RANDOM_CONTRAST_PLATES = (:T1, :T2, :PD)

# --- small helpers --------------------------------------------------------

# Scalar "sample spec": distribution, constant, or single-arg function f(rng).
# NOT the path for full (rng, d, ctx) property functions — those are dispatched
# by `_sample_property` in the declarative DSL.
_sample(rng::AbstractRNG, x) = rand(rng, x)
_sample(rng::AbstractRNG, x::Real) = Float64(x)
_sample(rng::AbstractRNG, f::Function) = f(rng)

# Sample an integer count from a fixed Int, a range, or a distribution-like spec.
_sample_count(::AbstractRNG, n::Integer) = Int(n)
_sample_count(rng::AbstractRNG, r::AbstractUnitRange) = Int(rand(rng, r))
_sample_count(rng::AbstractRNG, x) = round(Int, _sample(rng, x))

# Stable per-sphere index parsed from a label such as `:T1_14` -> 14.
function _label_index(label::Symbol)
    s = String(label)
    i = findlast('_', s)
    i === nothing && return 0
    parse(Int, s[(i + 1):end])
end

# Iterate plate keys in a deterministic, canonical order so RNG consumption is
# reproducible regardless of `Dict` hashing.
function _ordered_plates(keys_)
    ks = collect(keys_)
    canon = [p for p in _RANDOM_CONTRAST_PLATES if p in ks]
    extra = sort!([p for p in ks if !(p in _RANDOM_CONTRAST_PLATES)]; by = String)
    vcat(canon, extra)
end

# Reconstruct an (immutable) `PhantomConfig` with some fields overridden.
function _phantomconfig_with(cfg::PhantomConfig; kwargs...)
    overrides = Dict{Symbol,Any}(kwargs)
    PhantomConfig(; (f => get(overrides, f, getfield(cfg, f))
                     for f in fieldnames(PhantomConfig))...)
end

# Same per-spin noise but with sphere dropout disabled.
_augment_without_dropout(a::AugmentConfig) = AugmentConfig(
    T1_sigma_rel      = a.T1_sigma_rel,
    T2_sigma_rel      = a.T2_sigma_rel,
    PD_sigma_abs      = a.PD_sigma_abs,
    position_sigma_mm = a.position_sigma_mm,
    B0_sigma_Hz       = a.B0_sigma_Hz,
    drop_sphere_p     = 0.0,
)

# Clear the builder's own selection/dropout knobs (`drop_sphere_p`,
# `keep_sphere_labels`, `drop_sphere_labels`) while preserving per-spin noise.
# The random pipeline is the sole selection authority, so these must not act on
# either the nominal descriptors or the final build (e.g. randomly dropping the
# deterministic fiducial plate). `kwargs` lets callers add further overrides.
function _without_selection_knobs(cfg::PhantomConfig; kwargs...)
    _phantomconfig_with(cfg;
        augment            = _augment_without_dropout(cfg.augment),
        keep_sphere_labels = nothing,
        drop_sphere_labels = Symbol[],
        kwargs...)
end

# Base used to *generate nominal descriptors*: selection/dropout disabled so the
# `sphere_selector` is the only thing choosing spheres.
_clean_base_for_nominal(base::PhantomConfig) = _without_selection_knobs(base)

# Pure, stable function of the episode seed -- NOT derived from the episode RNG
# after sampling, so adding/reordering sampler draws never shifts build noise.
# Use explicit SplitMix64-style integer mixing rather than Base.hash, whose exact
# output is not a package-level reproducibility contract.
function _mix_seed64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27) * 0x94d049bb133111eb
    xor(x, x >> 31)
end

_episode_build_seed(episode_seed::Integer) =
    Int(_mix_seed64(xor(reinterpret(UInt64, Int64(episode_seed)),
                        0x6d72737068627569)) & UInt64(typemax(Int)))

# --- public types ---------------------------------------------------------

"""
    RandomPhantomConfig(; base, sphere_selector, material_sampler, pose_sampler,
                          rng_seed, rho_bounds, enforce_t2s_le_t2, validate)

A distribution over deterministic [`PhantomConfig`](@ref)s. Sampler fields are
duck-typed (`Distributions.jl` objects, constants, closures, or callable
structs). `base` supplies the deterministic build settings and the nominal
descriptors; it should leave selection/dropout off (the sampler disables them
internally when generating nominal descriptors).

See [`sample_phantom_config`](@ref), [`sample_phantom`](@ref).
"""
Base.@kwdef struct RandomPhantomConfig
    base::PhantomConfig          = PhantomConfig()
    sphere_selector              = nothing
    material_sampler             = nothing
    pose_sampler                 = nothing
    rng_seed::Int                = 0
    rho_bounds::Tuple{Float64,Float64} = (0.0, 1.0)
    enforce_t2s_le_t2::Bool      = true
    validate::Bool               = true
end

"""
    RandomPhantomEpisode(cfg, truth)

One sampled episode: `cfg` is an ordinary deterministic `PhantomConfig` (safe to
expose to the agent via [`build_phantom`](@ref)), `truth` is the hidden
descriptor-level, pre-augment record for evaluation. Do not expose `truth` to
the agent during training.
"""
struct RandomPhantomEpisode
    cfg::PhantomConfig
    truth::NamedTuple
end

# --- material result handling + validation --------------------------------

# Apply a material-sampler result (a `SphereDescriptor`, a `NamedTuple`/`Dict`
# of property overrides, or `nothing`) to a nominal descriptor.
function _descriptor_with_material(d::SphereDescriptor, result)
    result === nothing && return d
    result isa SphereDescriptor && return result
    # NamedTuple or AbstractDict: both support haskey/get on Symbol keys.
    T1 = Float64(get(result, :T1, d.T1))
    T2 = Float64(get(result, :T2, d.T2))
    # T2s default rule: explicit T2s wins; else track a newly-set T2; else keep
    # nominal T2s. Nothing downstream clamps T2s unconditionally.
    T2s = if haskey(result, :T2s)
        Float64(result[:T2s])
    elseif haskey(result, :T2)
        T2
    else
        d.T2s
    end
    ρ       = Float64(get(result, :ρ, d.ρ))
    delta_w = Float64(get(result, :delta_w, d.delta_w))
    SphereDescriptor(d.centre, d.radius, ρ, T1, T2, T2s, delta_w, d.label)
end

function _validate_material(d::SphereDescriptor, plate::Symbol, rpcfg::RandomPhantomConfig)
    rpcfg.validate || return d
    fail(msg) = error("Invalid sampled material for sphere $(d.label) on plate $plate:\n$msg")
    (isfinite(d.T1) && d.T1 > 0)  || fail("T1 must be finite and > 0, got $(d.T1)")
    (isfinite(d.T2) && d.T2 > 0)  || fail("T2 must be finite and > 0, got $(d.T2)")
    (isfinite(d.T2s) && d.T2s > 0) || fail("T2s must be finite and > 0, got $(d.T2s)")
    if rpcfg.enforce_t2s_le_t2
        # small relative tolerance for the T2s == T2 default case
        (d.T2s <= d.T2 * (1 + 1e-9)) || fail("T2s must be <= T2, got T2s=$(d.T2s), T2=$(d.T2)")
    end
    (isfinite(d.ρ) && d.ρ >= 0) || fail("ρ must be finite and >= 0, got $(d.ρ)")
    lo, hi = rpcfg.rho_bounds
    (lo <= d.ρ <= hi) || fail("ρ must be in [$lo, $hi], got $(d.ρ)")
    isfinite(d.delta_w) || fail("delta_w must be finite, got $(d.delta_w)")
    d
end

# Dispatch a material sampler. The generic fallback treats it as a callable
# `(rng, d, ctx) -> result`; the declarative sampler has its own method below.
_material_result(sampler, rng::AbstractRNG, d::SphereDescriptor, ctx) = sampler(rng, d, ctx)

# --- sphere selection -----------------------------------------------------

# Pick exactly k descriptors (without replacement), preserving original order.
function _select_k(rng::AbstractRNG, descs::AbstractVector{SphereDescriptor}, k::Integer)
    n = length(descs)
    k >= n && return copy(descs)
    k <= 0 && return SphereDescriptor[]
    idx = sort!(randperm(rng, n)[1:k])
    descs[idx]
end

function _nominal_descriptors_by_plate(base::PhantomConfig, plates)
    clean = _clean_base_for_nominal(base)
    Dict{Symbol,Vector{SphereDescriptor}}(p => sphere_descriptors(p, clean) for p in plates)
end

# Sample a pooled count across all contrast plates, then redistribute by plate.
function _select_pooled(rng::AbstractRNG, by_plate, spec)
    pool = Tuple{Symbol,SphereDescriptor}[]
    for p in _ordered_plates(keys(by_plate)), d in by_plate[p]
        push!(pool, (p, d))
    end
    k = clamp(_sample_count(rng, spec), 0, length(pool))
    idx = sort!(randperm(rng, length(pool))[1:k])
    out = Dict{Symbol,Vector{SphereDescriptor}}(p => SphereDescriptor[] for p in keys(by_plate))
    for i in idx
        p, d = pool[i]
        push!(out[p], d)
    end
    out
end

function _apply_sphere_selector(rng::AbstractRNG, selector, by_plate, base::PhantomConfig)
    if selector === nothing
        return Dict{Symbol,Vector{SphereDescriptor}}(p => copy(d) for (p, d) in by_plate)
    elseif selector isa Integer || selector isa AbstractUnitRange
        return _select_pooled(rng, by_plate, selector)
    elseif selector isa AbstractDict
        out = Dict{Symbol,Vector{SphereDescriptor}}()
        for p in _ordered_plates(keys(by_plate))
            spec = get(selector, p, nothing)
            out[p] = spec === nothing ? copy(by_plate[p]) :
                     _select_k(rng, by_plate[p], _sample_count(rng, spec))
        end
        return out
    else
        return selector(rng, by_plate, base)   # low-level callable contract
    end
end

# --- pose sampling --------------------------------------------------------

function _apply_pose(rng::AbstractRNG, sampler, base::PhantomConfig)
    sampler === nothing && return (base.rotation, base.translation_mm)
    res = sampler(rng, base)
    rotation       = get(res, :rotation, base.rotation)
    translation_mm = get(res, :translation_mm, base.translation_mm)
    (NTuple{3,Float64}(rotation), NTuple{3,Float64}(translation_mm))
end

# --- top-level sampling ---------------------------------------------------

"""
    sample_phantom_config(rpcfg; rng_seed = rpcfg.rng_seed) -> RandomPhantomEpisode

Sample one deterministic `PhantomConfig` from `rpcfg`. The pipeline is: generate
nominal contrast descriptors → select active spheres → sample materials →
validate → sample pose → assemble a deterministic config (randomised contrast
plates moved into `custom_sphere_descriptors`, a per-episode build seed) and a
hidden `truth` record.
"""
function sample_phantom_config(rpcfg::RandomPhantomConfig; rng_seed::Integer = rpcfg.rng_seed)
    episode_seed = Int(rng_seed)
    rng = Random.MersenneTwister(episode_seed)
    base = rpcfg.base

    contrast_plates = [p for p in base.include_plates if p in _RANDOM_CONTRAST_PLATES]
    nominal_by_plate = _nominal_descriptors_by_plate(base, contrast_plates)

    selected_by_plate = _apply_sphere_selector(rng, rpcfg.sphere_selector, nominal_by_plate, base)

    # Keep the selected nominal (pre-material) descriptors per plate. Source
    # identity (label/index) for RL masks comes from these, NOT from the sampled
    # descriptors — a material sampler may return a full SphereDescriptor that
    # relabels the sphere, which would corrupt the parsed index.
    selected_by_plate_sorted = Dict{Symbol,Vector{SphereDescriptor}}()
    sampled_by_plate = Dict{Symbol,Vector{SphereDescriptor}}()
    for plate in _ordered_plates(keys(selected_by_plate))
        # Sort by stable label index so RNG draws map to spheres deterministically.
        descs = sort(selected_by_plate[plate]; by = d -> _label_index(d.label))
        selected_by_plate_sorted[plate] = descs
        out = SphereDescriptor[]
        for d in descs
            ctx = (; plate,
                     index               = _label_index(d.label),
                     label               = d.label,
                     field               = base.field,
                     serial_number_class = base.serial_number_class,
                     cfg                 = base)
            result = rpcfg.material_sampler === nothing ? nothing :
                     _material_result(rpcfg.material_sampler, rng, d, ctx)
            nd = _descriptor_with_material(d, result)
            _validate_material(nd, plate, rpcfg)
            push!(out, nd)
        end
        sampled_by_plate[plate] = out
    end

    ordered = _ordered_plates(keys(sampled_by_plate))
    sampled_descs = SphereDescriptor[]
    for plate in ordered
        append!(sampled_descs, sampled_by_plate[plate])
    end

    rotation, translation_mm = _apply_pose(rng, rpcfg.pose_sampler, base)

    # Every randomisable contrast plate that was a *candidate* leaves
    # `include_plates`: a contrast plate is either represented by sampled custom
    # descriptors or absent — never silently regenerated deterministically.
    # Non-contrast plates (`:water`, `:fiducials`) stay deterministic.
    drop_from_include = union(Set(contrast_plates), keys(sampled_by_plate))
    new_include = [p for p in base.include_plates if !(p in drop_from_include)]
    # Preserve any pre-existing deterministic custom spheres alongside sampled ones.
    all_custom = vcat(base.custom_sphere_descriptors, sampled_descs)
    build_seed = _episode_build_seed(episode_seed)
    cfg = _without_selection_knobs(base;
        include_plates            = new_include,
        custom_sphere_descriptors = all_custom,
        rotation                  = rotation,
        translation_mm            = translation_mm,
        rng_seed                  = build_seed)

    # Source identity from the selected nominal descriptors (pre-material).
    active_indices_by_plate = Dict{Symbol,Vector{Int}}(
        p => sort!([_label_index(d.label) for d in descs])
        for (p, descs) in selected_by_plate_sorted)
    active_labels = Symbol[]
    for plate in ordered
        append!(active_labels, (d.label for d in selected_by_plate_sorted[plate]))
    end

    truth = (;
        descriptors_nominal     = nominal_by_plate,
        descriptors_sampled     = sampled_by_plate,
        active_labels,
        active_indices_by_plate,
        rotation,
        translation_mm,
        episode_seed,
        build_seed)

    RandomPhantomEpisode(cfg, truth)
end

"""
    sample_phantom(rpcfg; rng_seed = rpcfg.rng_seed) -> (; phantom, cfg, truth)

Convenience wrapper that also voxelises the sampled config via
[`build_phantom`](@ref).
"""
function sample_phantom(rpcfg::RandomPhantomConfig; rng_seed::Integer = rpcfg.rng_seed)
    episode = sample_phantom_config(rpcfg; rng_seed)
    phantom = build_phantom(episode.cfg)
    (; phantom, cfg = episode.cfg, truth = episode.truth)
end

"""
    eval_episodes(rpcfg, seeds) -> Vector{RandomPhantomEpisode}

Materialise a fixed evaluation pool from a held-out collection of seeds. Keep the
training seeds disjoint from `seeds` so the agent is never evaluated on a
configuration it trained on.
"""
eval_episodes(rpcfg::RandomPhantomConfig, seeds) =
    [sample_phantom_config(rpcfg; rng_seed = s) for s in seeds]

# --- Phase 1 convenience samplers -----------------------------------------

"""
    FixedPose(; rotation = nothing, translation_mm = nothing)

Pose sampler that keeps a fixed pose. `nothing` fields fall back to the base
config's pose.
"""
Base.@kwdef struct FixedPose
    rotation::Union{Nothing,NTuple{3,Float64}}       = nothing
    translation_mm::Union{Nothing,NTuple{3,Float64}} = nothing
end
(p::FixedPose)(::AbstractRNG, base::PhantomConfig) =
    (; rotation = something(p.rotation, base.rotation),
       translation_mm = something(p.translation_mm, base.translation_mm))

"""
    InPlanePoseSampler(; rotation_sigma_rad = 0.0, translation_sigma_mm = 0.0)

Gaussian in-plane pose: rotation about the slice normal (the Euler-Z component,
matching the default `slice_normal = (0,0,1)`) plus x/y translation. Avoids
out-of-plane tilt / z-translation that would move spheres out of a thin axial
slab.
"""
Base.@kwdef struct InPlanePoseSampler
    rotation_sigma_rad::Float64   = 0.0
    translation_sigma_mm::Float64 = 0.0
end
function (p::InPlanePoseSampler)(rng::AbstractRNG, ::PhantomConfig)
    γ  = p.rotation_sigma_rad * randn(rng)
    tx = p.translation_sigma_mm * randn(rng)
    ty = p.translation_sigma_mm * randn(rng)
    (; rotation = (0.0, 0.0, γ), translation_mm = (tx, ty, 0.0))
end

"""
    GaussianEulerPose(; rotation_sigma_rad = 0.0, translation_sigma_mm = 0.0)

Gaussian perturbation of all three Euler angles and all three translation axes.
Note this is *not* a uniform `SO(3)` orientation; for true uniform orientation a
dedicated `UniformSO3PoseSampler` (random unit quaternion) should be added later.
"""
Base.@kwdef struct GaussianEulerPose
    rotation_sigma_rad::Float64   = 0.0
    translation_sigma_mm::Float64 = 0.0
end
function (p::GaussianEulerPose)(rng::AbstractRNG, ::PhantomConfig)
    σr, σt = p.rotation_sigma_rad, p.translation_sigma_mm
    (; rotation = (σr * randn(rng), σr * randn(rng), σr * randn(rng)),
       translation_mm = (σt * randn(rng), σt * randn(rng), σt * randn(rng)))
end

"""
    E2SphereSelector(; subset_size, forced_indices = Int[], plate = :T1)

Reproduces the E2 policy: select `subset_size` spheres from one plate, always
including `forced_indices` (by stable label index). Other plates are dropped.
"""
Base.@kwdef struct E2SphereSelector
    subset_size::Int
    forced_indices::Vector{Int} = Int[]
    plate::Symbol               = :T1
end
function (s::E2SphereSelector)(rng::AbstractRNG, by_plate, ::PhantomConfig)
    descs = get(by_plate, s.plate, SphereDescriptor[])
    forced = SphereDescriptor[d for d in descs if _label_index(d.label) in s.forced_indices]
    rest   = SphereDescriptor[d for d in descs if !(_label_index(d.label) in s.forced_indices)]
    need = max(0, s.subset_size - length(forced))
    selected = vcat(forced, _select_k(rng, rest, need))
    sort!(selected; by = d -> _label_index(d.label))
    Dict{Symbol,Vector{SphereDescriptor}}(s.plate => selected)
end

"""
    SphereCountPerPlate(:T1 => 6:14, :T2 => 0:14, ...)

Selector that samples a per-plate sphere count (Int, range, or distribution).
Plates without an entry keep all their spheres. Named to avoid confusion with the
material-sampler container [`PerPlate`](@ref).
"""
struct SphereCountPerPlate
    counts::Dict{Symbol,Any}
end
SphereCountPerPlate(pairs::Pair...) = SphereCountPerPlate(Dict{Symbol,Any}(pairs))
function (s::SphereCountPerPlate)(rng::AbstractRNG, by_plate, ::PhantomConfig)
    out = Dict{Symbol,Vector{SphereDescriptor}}()
    for p in _ordered_plates(keys(by_plate))
        spec = get(s.counts, p, nothing)
        out[p] = spec === nothing ? copy(by_plate[p]) :
                 _select_k(rng, by_plate[p], _sample_count(rng, spec))
    end
    out
end

"""
    RatioPreservingLogNormalT1(sigma_rel)

Material sampler that jitters T1 log-normally (multiplicative,
`T1 * exp(sigma_rel * randn)`) and derives T2 from each sphere's nominal
`T2/T1` ratio. Matches the E2 default policy.
"""
struct RatioPreservingLogNormalT1
    sigma_rel::Float64
end
function (m::RatioPreservingLogNormalT1)(rng::AbstractRNG, d::SphereDescriptor, _ctx)
    T1 = d.T1 * exp(m.sigma_rel * randn(rng))
    T2 = T1 * d.T2 / d.T1
    (; T1, T2, T2s = T2)
end

# --- Phase 2: declarative material DSL ------------------------------------

"""
    PerLabel(:T1_4 => spec, ...; default = nothing)

Container property spec that branches on the sphere's exact label.
"""
struct PerLabel
    map::Dict{Symbol,Any}
    default::Any
end
PerLabel(pairs::Pair...; default = nothing) = PerLabel(Dict{Symbol,Any}(pairs), default)

"""
    PerPlate(:T1 => spec, ...; default = nothing)

Container property spec that branches on the sphere's plate.
"""
struct PerPlate
    map::Dict{Symbol,Any}
    default::Any
end
PerPlate(pairs::Pair...; default = nothing) = PerPlate(Dict{Symbol,Any}(pairs), default)

"""
    PerSphere([spec, spec, ...]; default = nothing)
    PerSphere(Dict(4 => spec, 14 => spec); default = nothing)

Container property spec that branches on the sphere's **stable label index**
(`:T1_14` -> 14), invariant under subset selection. The dense (`Vector`) form may
be shorter than the plate's sphere count; the sparse (`Dict`) form may omit
indices. A missing index falls through to `default`, then to the nominal value.
"""
struct PerSphere
    map::Dict{Int,Any}
    default::Any
end
PerSphere(v::AbstractVector; default = nothing) =
    PerSphere(Dict{Int,Any}(i => v[i] for i in eachindex(v)), default)
PerSphere(d::AbstractDict; default = nothing) =
    PerSphere(Dict{Int,Any}(d), default)

"""
    PreserveNominalRatio(dst, src)

Derived leaf for property `dst`: `dst = sampled[src] * nominal[dst] / nominal[src]`.
Requires `src` to be sampled (or falls back to nominal `src`).
"""
struct PreserveNominalRatio
    dst::Symbol
    src::Symbol
end

"""
    ScaledFrom(src, multiplier)

Derived leaf: `value = _sample(rng, multiplier) * sampled[src]`. `multiplier` is a
scalar sample spec (distribution / constant / single-arg `f(rng)`).
"""
struct ScaledFrom
    src::Symbol
    multiplier::Any
end

"""
    MaterialDistributionSampler(; T1, T2, T2s, ρ, delta_w)

Declarative material sampler. Each field is a property spec resolved recursively
per sphere (containers `PerLabel`/`PerPlate`/`PerSphere`; leaves are constants,
distributions, `(rng, d, ctx)` functions, or the derived helpers
[`PreserveNominalRatio`](@ref) / [`ScaledFrom`](@ref)). Properties are evaluated
in per-sphere topological order with cycle detection.
"""
Base.@kwdef struct MaterialDistributionSampler
    T1      = nothing
    T2      = nothing
    T2s     = nothing
    ρ       = nothing
    delta_w = nothing
end

const _MATERIAL_PROPS = (:ρ, :T1, :T2, :T2s, :delta_w)

# Recursively resolve a property spec to a leaf for one sphere. `nothing` means
# "keep nominal".
_resolve_property_sampler(spec, ::SphereDescriptor, _ctx) = spec
_resolve_property_sampler(::Nothing, ::SphereDescriptor, _ctx) = nothing
_resolve_property_sampler(s::PerLabel, d::SphereDescriptor, ctx) =
    _resolve_property_sampler(get(s.map, ctx.label, s.default), d, ctx)
_resolve_property_sampler(s::PerPlate, d::SphereDescriptor, ctx) =
    _resolve_property_sampler(get(s.map, ctx.plate, s.default), d, ctx)
_resolve_property_sampler(s::PerSphere, d::SphereDescriptor, ctx) =
    _resolve_property_sampler(get(s.map, ctx.index, s.default), d, ctx)

# Source properties a leaf depends on (only derived helpers create dependencies).
_leaf_deps(leaf::PreserveNominalRatio) = (leaf.src,)
_leaf_deps(leaf::ScaledFrom)           = (leaf.src,)
_leaf_deps(_leaf)                      = ()

function _topo_order(leaves::AbstractDict, d::SphereDescriptor, ctx)
    order = Symbol[]
    state = Dict{Symbol,Int}()   # 1 = in-progress, 2 = done
    function visit(p, path)
        st = get(state, p, 0)
        st == 2 && return
        if st == 1
            cyc = join(vcat(path, p), " -> ")
            error("Cyclic material property dependency for sphere $(d.label) on " *
                  "plate $(ctx.plate): $cyc")
        end
        state[p] = 1
        for s in _leaf_deps(leaves[p])
            haskey(leaves, s) && visit(s, vcat(path, p))
        end
        state[p] = 2
        push!(order, p)
    end
    # Visit in fixed _MATERIAL_PROPS order (not Dict-iteration order) so the
    # resulting evaluation order — and therefore the RNG draw order — is
    # reproducible and independent of Dict hashing.
    for p in _MATERIAL_PROPS
        haskey(leaves, p) && visit(p, Symbol[])
    end
    order
end

function _eval_leaf(leaf, rng::AbstractRNG, d::SphereDescriptor, ctx,
                    sampled::AbstractDict, nominal)
    if leaf isa PreserveNominalRatio
        srcv = get(sampled, leaf.src, getfield(nominal, leaf.src))
        return Float64(srcv * getfield(nominal, leaf.dst) / getfield(nominal, leaf.src))
    elseif leaf isa ScaledFrom
        srcv = get(sampled, leaf.src, getfield(nominal, leaf.src))
        return Float64(_sample(rng, leaf.multiplier) * srcv)
    elseif leaf isa Function
        return Float64(leaf(rng, d, ctx))      # full (rng, d, ctx) property function
    else
        return Float64(_sample(rng, leaf))     # distribution or constant
    end
end

function _material_result(s::MaterialDistributionSampler, rng::AbstractRNG,
                          d::SphereDescriptor, ctx)
    leaves = Dict{Symbol,Any}()
    for p in _MATERIAL_PROPS
        leaf = _resolve_property_sampler(getfield(s, p), d, ctx)
        leaf === nothing && continue
        leaves[p] = leaf
    end
    isempty(leaves) && return nothing
    nominal = (; T1 = d.T1, T2 = d.T2, T2s = d.T2s, ρ = d.ρ, delta_w = d.delta_w)
    sampled = Dict{Symbol,Float64}()
    for p in _topo_order(leaves, d, ctx)
        sampled[p] = _eval_leaf(leaves[p], rng, d, ctx, sampled, nominal)
    end
    sampled   # AbstractDict result; flows through _descriptor_with_material
end
