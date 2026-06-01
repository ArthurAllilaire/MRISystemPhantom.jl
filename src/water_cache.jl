# Cached-Koma-water model — simulate the homogeneous background water ONCE per
# geometry (across a small grid of excitation flip angles), then add its k-space
# contribution analytically at every step instead of full-Bloch simulating ~89%
# of the phantom's spins each time.
#
# Why this works (to the forward-model floor): water is one material, so the raw
# multi-shot k-space factorises as
#
#     S_water(row k, :) = Mz_shot[k](TI,TR,α) · sin(α) · W_α[k, :]
#
# where `W_α[k,:]` is a geometric template (spin positions, T2 echo decay at the
# reference TE, static B0 phase). KomaMRI is linear in spins, so the full k-space
# is exactly spheres_kspace + water_kspace — we Bloch-sim only the spheres and add
# the rescaled template for the water.
#
# "Per-line" correction: each k-space row k is acquired on shot k and carries that
# shot's transient Mz[k]; we rescale each row by its OWN shot's transient. Per-shot
# Mz[k] is recovered by finite-differencing the cumulative mean of the existing
# forward model `transient_mz_at_excite_npe` — no new recurrence.
#
# α-DEPENDENCE — why a single reference is NOT enough (measured 2026-05-27):
# the analytic per-shot transient mis-scales with α (Koma/analytic up to ~1.4, and
# row-dependent) AND the geometric template `W_α` is itself ~7–21% α-dependent. A
# single 90° reference run on a 30° schedule gives 9% mean / 61% max T1 error vs
# Bloch. BUT at a MATCHED α (reference α == step α) the template is exact (~1e-8
# relerr) and generalises well over TI/TR. So we cache a BANK of α-matched
# templates on a grid and, for an off-grid α, evaluate the two bracketing templates
# *each at its own α* (where each is accurate) and linearly blend in α. We never
# interpolate a template and then rescale it to a different α analytically — that
# would reintroduce the bad α-scaling.
#
# Validated by scripts/hybrid_water_run.jl (TI/TR generalisation at fixed α) and
# scripts/cached_water_validation.jl (α-bank, B0σ=5 Hz, T1 fits vs Bloch). Used by
# src/rl/e2.jl and the hybrid-water harness.

"""
    CachedWaterModel

Precomputed background-water k-space templates for one geometry, one per excitation
flip angle on `α_grid`. Build with [`build_cached_water_model`](@ref); evaluate per
step with [`cached_water_ksp`](@ref) (which interpolates across `α_grid`).

Fields:
- `α_grid` : excitation flip angles [rad], strictly ascending. A length-1 grid is a
  single α-matched reference (exact only when evaluated at that α).
- `w_lines` : geometric `Npe×Nfe` template per `α_grid` entry (the reference water
  k-space at that α with its reference per-shot transient and `sin α` divided out).
- `T1_water`, `T2_water` : water relaxation [s] (read off the water phantom).
- `Npe`, `Nfe` : k-space dimensions.
- `TI_ref`, `TR_ref`, `TE_ref` : reference operating point. `TE_ref` fixes the T2
  echo decay baked into the templates; per-step TE differences are corrected in
  [`cached_water_ksp`](@ref).
"""
struct CachedWaterModel
    α_grid   :: Vector{Float64}
    w_lines  :: Vector{Matrix{ComplexF32}}
    T1_water :: Float64
    T2_water :: Float64
    Npe      :: Int
    Nfe      :: Int
    TI_ref   :: Float64
    TR_ref   :: Float64
    TE_ref   :: Float64
end


"""
    build_cached_water_model(water_phantom, scanner; FOV, Nfe, Npe,
        α_grid = [deg2rad(90.0)], TI_ref=0.10, TR_ref=5.0, TE_ref=0.020,
        use_gpu=false) -> CachedWaterModel

Simulate `water_phantom` once per `α_grid` entry (noiseless) at the reference TI/TR
and build the geometric per-line template at each α. `water_phantom` must contain
only the homogeneous background-water spins (e.g. from [`build_dry_and_water`](@ref));
its `T1`/`T2` are read off the first spin. `length(α_grid)` simulations total — for
a fixed-α run pass a single-element grid (exact); for a learned/variable-α run pass
a grid spanning the action's α range (interpolated per step).
"""
function build_cached_water_model(water_phantom, scanner;
                                  FOV::Real, Nfe::Int, Npe::Int,
                                  α_grid::AbstractVector{<:Real} = [deg2rad(90.0)],
                                  TI_ref::Real = 0.10, TR_ref::Real = 5.0,
                                  TE_ref::Real = 0.020, use_gpu::Bool = false)
    length(water_phantom.x) > 0 ||
        error("build_cached_water_model: water_phantom has no spins")
    grid = Float64.(collect(α_grid))
    (length(grid) >= 1 && issorted(grid) && allunique(grid)) ||
        error("build_cached_water_model: α_grid must be non-empty, sorted, unique")
    T1_water = Float64(water_phantom.T1[1])
    T2_water = Float64(water_phantom.T2[1])

    w_lines = Vector{Matrix{ComplexF32}}(undef, length(grid))
    for (g, α) in enumerate(grid)
        seq = Suppressor.@suppress ir_se_2d_sequence(
            Float64(TI_ref), Float64(TE_ref), Float64(TR_ref);
            α_exc = α, FOV = FOV, Nfe = Nfe, Npe = Npe)
        raw = Suppressor.@suppress simulate(
            water_phantom, seq, scanner;
            sim_params = Dict{String,Any}("gpu" => use_gpu))
        ksp_ref = raw_to_kspace(raw, Npe, Nfe)
        mz_ref = transient_mz_per_shot(T1_water, TI_ref, TR_ref, π, α, Npe)
        sin_α = sin(α)
        w = similar(ksp_ref)
        @inbounds for k in 1:Npe
            w[k, :] = ksp_ref[k, :] ./ ComplexF32(mz_ref[k] * sin_α)
        end
        w_lines[g] = w
    end

    CachedWaterModel(grid, w_lines, T1_water, T2_water, Npe, Nfe,
                     Float64(TI_ref), Float64(TR_ref), Float64(TE_ref))
end

# Evaluate the template at grid index `g` at ITS OWN α (where it is accurate) and
# the target TI/TR/TE: rescale each row by its shot transient · sin α · T2 echo.
function _eval_grid(m::CachedWaterModel, g::Int, TI::Real, TR::Real, TE::Real)
    α = m.α_grid[g]
    mz = transient_mz_per_shot(m.T1_water, TI, TR, π, α, m.Npe)
    sin_α = sin(α)
    te_fac = exp(-(Float64(TE) - m.TE_ref) / m.T2_water)
    out = similar(m.w_lines[g])
    @inbounds for k in 1:m.Npe
        out[k, :] = m.w_lines[g][k, :] .* ComplexF32(mz[k] * sin_α * te_fac)
    end
    out
end

"""
    cached_water_ksp(m::CachedWaterModel, TI, TR, α_exc, TE) -> Matrix{ComplexF32}

Background-water k-space at `(TI, TR, α_exc, TE)`. The two `α_grid` templates
bracketing `α_exc` are each evaluated at their own α (per-line shot transient ·
sin α · spin-echo T2 correction `exp(-(TE−TE_ref)/T2_water)`) and linearly blended
in α. `α_exc` is clamped to the grid range. Add the result to the spheres-only
Bloch k-space to reconstruct the full image.
"""
function cached_water_ksp(m::CachedWaterModel, TI::Real, TR::Real,
                          α_exc::Real, TE::Real)
    grid = m.α_grid
    α = clamp(Float64(α_exc), grid[1], grid[end])
    if length(grid) == 1 || α <= grid[1]
        return _eval_grid(m, 1, TI, TR, TE)
    elseif α >= grid[end]
        return _eval_grid(m, length(grid), TI, TR, TE)
    end
    hi = searchsortedfirst(grid, α)          # first index with grid[hi] >= α
    if grid[hi] == α
        return _eval_grid(m, hi, TI, TR, TE)
    end
    lo = hi - 1
    w  = (α - grid[lo]) / (grid[hi] - grid[lo])
    c_lo = _eval_grid(m, lo, TI, TR, TE)
    c_hi = _eval_grid(m, hi, TI, TR, TE)
    @. c_lo = (1 - w) * c_lo + w * c_hi
    c_lo
end

# ── PhantomConfig / AugmentConfig copy-with-override (immutable @kwdef structs) ─
_config_with(cfg::T; kwargs...) where {T} =
    T(; (f => get(kwargs, f, getfield(cfg, f)) for f in fieldnames(T))...)

"""
    build_dry_and_water(cfg::PhantomConfig; water_B0_sigma_Hz=nothing)
        -> (dry::Phantom, water::Phantom)

Split a phantom into its spheres-only part and its background-water part, exploiting
that `build_phantom` lays spins out as `[spheres…, water…]` and that KomaMRI is
linear in spins. Builds the full phantom (with `:water`) and the spheres-only
phantom (without `:water`) under the *same* cfg/seed/pose, asserts the sphere-prefix
layout, and slices off the water block. The `water` phantom feeds
[`build_cached_water_model`](@ref); `dry` is Bloch-simulated each step.

`water_B0_sigma_Hz`, when given, overrides the B0 off-resonance σ of the water
phantom (the spheres in `dry` keep the cfg's augment). Set it to `0.0` to build a
coherent (off-resonance-free) water template: per-spin B0 makes the cached water
k-space TI-dependent in phase, which the analytic per-line rescale cannot track, so
the cache is only accurate at B0σ=0 (E2 keeps B0σ on the Bloch-simulated spheres).
B0 augment does not move spins, so the sphere-prefix layout still matches.
"""
function build_dry_and_water(cfg::PhantomConfig; water_B0_sigma_Hz = nothing)
    plates_dry = [p for p in cfg.include_plates if p != :water]
    dry = build_phantom(_config_with(cfg; include_plates = plates_dry))
    cfg_full = _config_with(cfg; include_plates = vcat(plates_dry, :water))
    if water_B0_sigma_Hz !== nothing
        cfg_full = _config_with(cfg_full;
            augment = _config_with(cfg_full.augment;
                                   B0_sigma_Hz = Float64(water_B0_sigma_Hz)))
    end
    full = build_phantom(cfg_full)
    n_dry = length(dry.x)
    length(full.x) >= n_dry ||
        error("build_dry_and_water: full phantom smaller than dry phantom")
    @views(full.x[1:n_dry]) ≈ dry.x ||
        error("build_dry_and_water: phantom layout is not [spheres…, water…]")
    mask = falses(length(full.x))
    mask[(n_dry+1):end] .= true
    water = full[mask]
    dry, water
end
