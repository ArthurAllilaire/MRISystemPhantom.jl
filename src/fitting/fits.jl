# Pure-Julia monoexponential fits used by the E0 baseline and the
# running-estimate feature of the later RL environments. No external
# optimiser dependency — T2 is closed-form in log space, T1 is a log-grid
# search over T1 with closed-form (A, B) at each T1.
# TODO: look into using PyQMRI for better algos
# REPO: https://gitlab.tugraz.at/ibi/mrirecon/software/PyQMRI
# paper: https://joss.theoj.org/papers/10.21105/joss.02727
# or matlab version: https://github.com/qMRLab/qMRLab


"""
    fit_t2_se(TEs, magnitudes) -> (T2, S0, residual)

Log-linear fit of `|S| = S0 · exp(−TE/T2)` to magnitude-only SE data.
Returns the T2 estimate (s), amplitude S0, and the RMS of the residuals
in log space. Points with non-positive magnitude are dropped.
"""
function fit_t2_se(TEs::AbstractVector{<:Real},
                   mags::AbstractVector{<:Real})
    length(TEs) == length(mags) || error("TEs and mags length mismatch")
    keep = mags .> 0
    count(keep) >= 2 || error("Need ≥2 positive samples for T2 fit")
    t = Float64.(TEs[keep])
    y = log.(Float64.(mags[keep]))
    X = hcat(ones(length(t)), -t)
    β = X \ y
    S0 = exp(β[1])
    slope = β[2]
    slope > 0 || error("Non-decaying signal; slope = $slope")
    T2 = 1 / slope
    resid = sqrt(sum((X * β .- y).^2) / length(y))
    (T2 = T2, S0 = S0, residual = resid)
end

"""
    fit_t1_ir(TIs, mags; T1_range = (5e-3, 5.0), n_grid = 400)

3-parameter magnitude-IR fit: `|S| = |A − B · exp(−TI/T1)|`. A log-spaced
grid over `T1_range` is scanned; at each candidate T1 the optimal (A, B)
are closed-form from a linear regression of signed-sign · mag on
(1, exp(−TI/T1)). The best SSE wins.

The signed-sign is recovered heuristically: samples with TI below the
null time (where the sign flips) are negated before fitting. With a
magnitude acquisition we can only tell by iterating — we try both sign
patterns per candidate T1 and keep the better one.

Returns `(T1, A, B, residual)`.
"""
function fit_t1_ir(TIs::AbstractVector{<:Real},
                   mags::AbstractVector{<:Real};
                   T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                   n_grid::Int = 400)
    length(TIs) == length(mags) || error("TIs and mags length mismatch")
    length(TIs) >= 3 || error("Need ≥3 samples for 3-param IR fit")

    ti = Float64.(TIs)
    y  = Float64.(mags)

    T1_candidates = exp.(range(log(T1_range[1]), log(T1_range[2]);
                               length = n_grid))

    best_sse = Inf
    best = (T1 = NaN, A = NaN, B = NaN)

    # We don't know the true sign pattern (magnitude data). For a monotone
    # |A − B·e^{-TI/T1}|, the sign flips at most once at the null
    # TI*. For each candidate T1, try every prefix-flip of the sorted-TI
    # data and keep the best.
    order = sortperm(ti)
    ti_sorted = ti[order]
    y_sorted  = y[order]
    n = length(ti)

    for T1 in T1_candidates
        e = exp.(-ti_sorted ./ T1)
        for k in 0:n               # flip the first k points
            signs = ones(n)
            signs[1:k] .= -1.0
            y_signed = signs .* y_sorted
            # y = A - B·e → linear regression on [1, -e]
            X = hcat(ones(n), -e)
            β = X \ y_signed
            A, B = β[1], β[2]
            resid = X * β .- y_signed
            sse = sum(abs2, resid)
            if sse < best_sse
                best_sse = sse
                best = (T1 = T1, A = A, B = B)
            end
        end
    end

    (T1 = best.T1, A = best.A, B = best.B,
     residual = sqrt(best_sse / length(ti)))
end

"""
    steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc) -> Mz

Steady-state longitudinal magnetisation (in units of M0) at the moment of
the excitation pulse, for an IR-prep block of the form

    [θ_inv inversion] → TI → [α_exc excitation, ADC] → (TR − TI recovery)

repeated indefinitely with **perfect transverse spoiling between TRs**.
Derivation: solve Mz_pre (just before inversion) from the recurrence

    Mz_at_TI = (1 − E1) + cos(θ_inv) · E1 · Mz_pre
    Mz_pre   = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI

with E1 = exp(−TI/T1), E2 = exp(−(TR−TI)/T1). For TR → ∞ this collapses
to the conventional non-steady-state form `1 − (1 − cos θ_inv)·exp(−TI/T1)`.
Special cases:
- θ_inv = π,  α_exc = π/2:  Mz = 1 − 2·exp(−TI/T1) + exp(−TR/T1)  (textbook IR)
- TR = ∞:                    Mz = 1 − (1 − cos θ_inv)·exp(−TI/T1)
"""
@inline function steady_state_mz_at_excite(T1::Real, TI::Real, TR::Real,
                                            θ_inv::Real, α_exc::Real)
    T1f = Float64(T1)
    E1  = exp(-Float64(TI) / T1f)
    if !isfinite(TR) || TR <= 0
        # TR → ∞ : full recovery between repetitions (Mz_pre = M0)
        # α_exc has no effect
        # on Mz_at_excite (it only matters via the sin(α_exc) projection
        # into Mxy, which the caller handles).
        return 1.0 - (1.0 - cos(θ_inv)) * E1
    end
    a   = cos(θ_inv)
    b   = cos(α_exc)
    E2  = exp(-(Float64(TR) - Float64(TI)) / T1f)
    num = (1 - E2) + b * E2 * (1 - E1)
    den = 1 - a * b * E1 * E2
    Mz_pre = num / den
    # returns Mz_at_TI
    return 1 - E1 + a * E1 * Mz_pre
end

"""
    transient_mz_per_shot(T1, TI, TR, θ_inv, α_exc, Npe) -> Vector{Float64}

Per-shot longitudinal magnetisation at excitation for each of the `Npe` shots
of an IR block starting from thermal equilibrium. `Mz[k]` weights k-space row
`k`. See [`transient_mz_at_excite_npe`](@ref) for the recurrence and limits.
"""
function transient_mz_per_shot(T1::Real, TI::Real, TR::Real, θ_inv::Real,
                               α_exc::Real, Npe::Int)
    Npe ≥ 1 || error("Npe must be ≥ 1")
    T1f = Float64(T1)
    E1  = exp(-Float64(TI) / T1f)
    a   = cos(Float64(θ_inv))
    out = Vector{Float64}(undef, Npe)
    if !isfinite(TR) || TR <= 0
        fill!(out, 1.0 - (1.0 - a) * E1)
        return out
    end
    b      = cos(Float64(α_exc))
    E2     = exp(-(Float64(TR) - Float64(TI)) / T1f)
    Mz_pre = 1.0
    @inbounds for k in 1:Npe
        Mz_at_TI = (1.0 - E1) + a * E1 * Mz_pre
        out[k]   = Mz_at_TI
        Mz_pre   = (1.0 - E2) + b * E2 * Mz_at_TI
    end
    out
end

"""
    transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe::Int) -> Mz

Mean Mz_at_excite across the `Npe` shots of an IR block that starts from
thermal equilibrium (`Mz_pre[1] = M0 = 1`). This is the closed form that
matches what `ir_se_2d_sequence` actually feeds the simulator: a sequence
of `Npe` IR-SE shots, each separated by TR, with k-space row k acquired
on shot k, reconstructed via IFFT (whose DC term ≈ mean over shots for a
small-pixel sphere). Discussion + derivation: `docs/T1_FIT_AND_KOMA_TESTS.md`
§7.5 / §19.5.1 of `EXPERT_REPORT.md`.

Recurrence:

    Mz_pre[1]    = 1
    for k = 1..Npe:
        Mz_at_TI[k]  = (1 − E1) + cos(θ_inv) · E1 · Mz_pre[k]
        Mz_pre[k+1]  = (1 − E2) + cos(α_exc) · E2 · Mz_at_TI[k]
    return mean(Mz_at_TI[1:Npe])

with E1 = exp(−TI/T1), E2 = exp(−(TR−TI)/T1). Limits:
- `Npe = 1`     ⇒ legacy TR-blind transient form `1 − (1 − cos θ_inv) e^{−TI/T1}`.
- `Npe → ∞`     ⇒ `steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc)`.
- `TR → ∞`      ⇒ `Mz_pre[k] ≡ 1` ⇒ pure transient (every shot from M0).
"""
@inline transient_mz_at_excite_npe(T1::Real, TI::Real, TR::Real, θ_inv::Real,
                                    α_exc::Real; Npe::Int) =
    mean(transient_mz_per_shot(T1, TI, TR, θ_inv, α_exc, Npe))

# ───────────────────────── fit_t1_generalized_ir helpers ─────────────────────
# Each helper owns one section of the generalised IR fit; the public function
# below is just orchestration. Boundaries chosen so the bootstrap σ branch can
# call `_scan_grid` directly on resampled data instead of inlining a near-
# duplicate loop, and so each σ estimator (asymptotic / profile / bootstrap) is
# independently readable.

# Build the per-grid-point forward-model closure used by the scan + Jacobian.
# Dispatches steady-state vs transient F1+ on `use_transient`.
function _build_predict(use_transient::Bool,
                        ti::Vector{Float64}, tr::Vector{Float64},
                        al::Vector{Float64}, ae::Vector{Float64},
                        Npe_i::Int)
    n = length(ti)
    function predict(T1)
        out = Vector{Float64}(undef, n)
        if use_transient
            @inbounds for k in 1:n
                out[k] = transient_mz_at_excite_npe(T1, ti[k], tr[k],
                                                     al[k], ae[k];
                                                     Npe = Npe_i)
            end
        else
            @inbounds for k in 1:n
                out[k] = steady_state_mz_at_excite(T1, ti[k], tr[k],
                                                    al[k], ae[k])
            end
        end
        return out
    end
    return predict
end

# Log-spaced T1 grid. With `T1_oracle` set, narrow to a ±oracle_band log-band
# around the truth — diagnostic only; tests for "is LM stuck in the wrong
# basin?" by collapsing the search to the right one.
function _t1_grid(T1_range::NTuple{2,<:Real}, n_grid::Int,
                  T1_oracle::Union{Nothing,Real}, oracle_band::Real)
    grid_lo, grid_hi = if T1_oracle === nothing
        (Float64(T1_range[1]), Float64(T1_range[2]))
    else
        T1o = Float64(T1_oracle)
        T1o > 0 || error("T1_oracle must be positive")
        b = Float64(oracle_band)
        b > 1 || error("oracle_band must be > 1 (e.g. 2.0 for ±1 octave)")
        (max(T1o / b, Float64(T1_range[1])),
         min(T1o * b, Float64(T1_range[2])))
    end
    return exp.(range(log(grid_lo), log(grid_hi); length = n_grid))
end

# Closed-form (A*, SSE) at one cached forward-model vector against data m_vec.
# Same formula for signed and magnitude paths — only the |·| changes.
@inline function _A_and_sse(y::Vector{Float64},
                             m_vec::AbstractVector{Float64}, signed::Bool)
    ay = signed ? y : abs.(y)
    den = sum(abs2, ay)
    den > 0 || return (NaN, Inf)
    A = sum(m_vec .* ay) / den
    r = A .* ay .- m_vec
    return (A, sum(abs2, r))
end

# Grid scan: returns the best (T1*, A*, SSE*, idx) and the full SSE-grid the
# profile-likelihood σ consumes. Re-used by the bootstrap σ on resampled data.
function _scan_grid(predict_cache::Vector{Vector{Float64}},
                    T1_candidates::Vector{Float64},
                    m_vec::AbstractVector{Float64}; signed::Bool)
    n_grid = length(predict_cache)
    sse_grid = fill(Inf, n_grid)
    best_sse = Inf
    best_T1  = NaN
    best_A   = NaN
    best_idx = 0
    @inbounds for i in 1:n_grid
        A, sse = _A_and_sse(predict_cache[i], m_vec, signed)
        sse_grid[i] = sse
        if sse < best_sse
            best_sse = sse
            best_T1  = T1_candidates[i]
            best_A   = A
            best_idx = i
        end
    end
    return (best_idx = best_idx, best_T1 = best_T1, best_A = best_A,
            best_sse = best_sse, sse_grid = sse_grid)
end

# σ²_eff = max(σ²_resid, σ²_floor) under the Fix A / Fix B rules.
# Fix A: at n ≤ 4 the residual variance has no statistical power, so σ²_resid
#        is set to +∞ implicitly by replacing the max with the floor outright
#        (else `max(Inf, floor) = Inf` would collapse σ even with an honest
#        floor supplied — `docs/T1_FIT_AND_KOMA_TESTS.md` §7.1).
# Fix B: `abs_noise_sigma` is preferred over the relative `noise_sigma`,
#        decoupling σ_T1 from the agent's data-RMS-via-action choice (§7.3).
function _sigma_eff(best_sse::Float64, n::Int, m::Vector{Float64},
                    abs_noise_sigma::Union{Nothing,Real},
                    noise_sigma::Union{Nothing,Real})
    σ²_floor = if abs_noise_sigma !== nothing
        Float64(abs_noise_sigma)^2
    elseif noise_sigma !== nothing
        rms_m = sqrt(sum(abs2, m) / n)
        (Float64(noise_sigma) * rms_m)^2
    else
        0.0
    end
    return n > 4 ? max(best_sse / (n - 2), σ²_floor) : σ²_floor
end

# Asymptotic σ_T1 from a central-difference Jacobian at the LM optimum.
# Magnitude model: model_k = A·|y_k|; ∂model/∂T1 ≈ A·sign(y_k)·(y_p−y_m)/(2h).
# Signed model:    model_k = A· y_k ; ∂model/∂T1 ≈ A·            (y_p−y_m)/(2h).
# Σ ≈ σ²_eff · (J^T J)^-1; we take the [T1, T1] entry.
function _sigma_asymptotic(predict, best_T1::Float64, best_A::Float64,
                            σ²_eff::Float64; signed::Bool)
    h   = max(1e-6, 1e-4 * best_T1)
    y0  = predict(best_T1)
    y_p = predict(best_T1 + h)
    y_m = predict(best_T1 - h)
    if signed
        dA  = y0
        dT1 = best_A .* (y_p .- y_m) ./ (2h)
    else
        sgn = sign.(y0)
        dA  = abs.(y0)
        dT1 = best_A .* sgn .* (y_p .- y_m) ./ (2h)
    end
    a11    = sum(abs2, dA)
    a22    = sum(abs2, dT1)
    a12    = sum(dA .* dT1)
    det_JtJ = a11 * a22 - a12^2
    (det_JtJ > 0 && σ²_eff > 0) || return NaN
    var_T1 = σ²_eff * a11 / det_JtJ
    return var_T1 > 0 ? sqrt(var_T1) : NaN
end

# Profile-likelihood σ: half-span of T1s whose SSE is within `σ²_eff` of
# SSE_min (asymptotic χ²₁ test, conf 0.683). Captures multimodal basins that
# asymptotic σ misses. §15 of `cr_explainer.md`.
function _sigma_profile(sse_grid::Vector{Float64},
                        T1_candidates::Vector{Float64},
                        best_sse::Float64, σ²_eff::Float64)
    σ²_eff > 0 || return NaN
    threshold = best_sse + σ²_eff
    pass_mask = sse_grid .≤ threshold
    any(pass_mask) || return NaN
    T1_lo = minimum(T1_candidates[pass_mask])
    T1_hi = maximum(T1_candidates[pass_mask])
    return (T1_hi - T1_lo) / 2
end

# Bootstrap σ: resample residuals at the LM optimum, refit each synthetic
# dataset via `_scan_grid`, return std of T1*_b. The right tool for
# wrong-basin multimodal-SSE failures — bootstrap samples can land in
# different basins under noise, and that spread is the honest σ.
function _sigma_bootstrap(predict_cache::Vector{Vector{Float64}},
                          T1_candidates::Vector{Float64},
                          m::Vector{Float64},
                          best_idx::Int, best_A::Float64;
                          signed::Bool, n_bootstrap::Int,
                          bootstrap_seed::Int)
    n = length(m)
    n >= 2 || return NaN
    y_at_best = predict_cache[best_idx]
    ay_at_best = signed ? y_at_best : abs.(y_at_best)
    fitted    = best_A .* ay_at_best
    residuals = m .- fitted

    rng_b = MersenneTwister(bootstrap_seed)
    T1_boots = Vector{Float64}(undef, n_bootstrap)
    @inbounds for b in 1:n_bootstrap
        idxs = rand(rng_b, 1:n, n)
        m_b  = fitted .+ residuals[idxs]
        scan = _scan_grid(predict_cache, T1_candidates, m_b; signed = signed)
        T1_boots[b] = scan.best_T1
    end
    valid = filter(isfinite, T1_boots)
    length(valid) >= 2 || return NaN
    μ      = sum(valid) / length(valid)
    var_T1 = sum((x - μ)^2 for x in valid) / (length(valid) - 1)
    return sqrt(var_T1)
end

"""
    fit_t1_generalized_ir(TIs, αs, mags; TRs, α_excs, Npe, T1_range, n_grid,
                          noise_sigma, abs_noise_sigma)

Fit T1 to (TI, α_inv, |S|) triples under an IR forward model

    |S| = |A · Mz_at_excite(T1; TI, TR, θ_inv, α_exc)|

`αs` is the **inversion / preparation** flip angle (e.g. π for canonical
IR, π/2 for SR). The optional `α_excs` gives the **excitation** flip
angle per sample; defaults to π/2 (so post-excitation Mz = 0). The
optional `TRs` gives the per-block repetition time; if omitted, each TR
is taken to be ∞.

`Npe` selects the forward model:
- `Npe === nothing` (default): steady-state IR (`steady_state_mz_at_excite`).
  Back-compat with the original generalized-IR fit.
- `Npe::Int`: finite-Npe transient ramp from M0=1
  (`transient_mz_at_excite_npe`). This matches what
  `ir_se_2d_sequence` actually does in E2 — see `EXPERT_REPORT.md` §19.5.1
  for why this matters under adaptive-action sequences.

A log-spaced T1 grid is scanned; at each candidate T1, the optimal |A|
is the closed-form ratio `sum(m · |y|) / sum(|y|²)`. Returns
`(T1, A, residual, T1_sigma)`.

`sigma_method` selects how `T1_sigma` is computed:
- `:asymptotic` (default): Cramér–Rao from Jacobian local at the LM minimum,
  `Σ ≈ σ²_eff · (J^T J)^-1`. Fast, smooth, but **over-confident on multimodal
  SSE** (the LM minimum is in one basin; asymptotic σ doesn't see other
  basins). Used by all pre-E2.5 callers.
- `:profile_likelihood`: span of T1s whose SSE is within `σ²_eff` of the
  minimum (asymptotic χ²₁ test, conf 0.683). Captures multimodal-SSE basins
  honestly — wide σ on degenerate fits, tight σ on well-determined ones. Use
  this for the σ-channel obs and any reported uncertainty after E2.5. See
  §15 of `cr_explainer.md` for the full derivation and §3 of `E2_5_PLAN.md`.

`σ²_eff = max(σ²_resid, σ²_floor)` where:
- σ²_resid = `best_sse / (n − 2)` for `n > 4`; **`Inf` for `n ≤ 4`**
  (Fix A: the residual variance has no statistical power below 5 samples
  with 2 free parameters, see `docs/T1_FIT_AND_KOMA_TESTS.md` §7.1).
- σ²_floor: pick **one** of the two paths.
  - `abs_noise_sigma` (Fix B): absolute noise σ in magnitude units. Used
    directly as the floor: `σ²_floor = abs_noise_sigma²`. Recommended —
    decouples σ_T1 from the agent's data-RMS-via-action choice
    (`docs/T1_FIT_AND_KOMA_TESTS.md` §7.3).
  - `noise_sigma`: legacy *relative* noise level (fraction of data RMS).
    `σ²_floor = (noise_sigma · rms(m))²`. Kept for back-compat.
  If both are passed, `abs_noise_sigma` wins.

Caller convention: `mags` should already be divided by sin(α_exc) per
sample (see `_e2_update_t1_estimates!`).

Degenerate at length < 2 — throws.
"""
function fit_t1_generalized_ir(TIs::AbstractVector{<:Real},
                               αs::AbstractVector{<:Real},
                               mags::AbstractVector{<:Real};
                               TRs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                               α_excs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                               Npe::Union{Nothing,Int} = nothing,
                               T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                               n_grid::Int = 300,
                               noise_sigma::Union{Nothing,Real} = nothing,
                               abs_noise_sigma::Union{Nothing,Real} = nothing,
                               sigma_method::Symbol = :asymptotic,
                               n_bootstrap::Int = 100,
                               bootstrap_seed::Int = 0,
                               signed::Bool = false,
                               T1_oracle::Union{Nothing,Real} = nothing,
                               oracle_band::Real = 2.0)
    n = length(TIs)
    n == length(αs) == length(mags) ||
        error("TIs, αs, mags must be same length")
    n >= 2 || error("Need ≥2 samples for generalized IR fit")
    TRs    === nothing || length(TRs)    == n || error("TRs length mismatch")
    α_excs === nothing || length(α_excs) == n || error("α_excs length mismatch")
    Npe    === nothing || Npe ≥ 1                 || error("Npe must be ≥ 1")
    sigma_method in (:asymptotic, :profile_likelihood, :bootstrap) ||
        error("sigma_method must be :asymptotic, :profile_likelihood, " *
              "or :bootstrap, got :$sigma_method")

    ti = Float64.(TIs)
    al = Float64.(αs)                                     # inversion angle
    m  = Float64.(mags)
    tr = TRs    === nothing ? fill(Inf, n) : Float64.(TRs)
    ae = α_excs === nothing ? fill(π/2, n) : Float64.(α_excs)

    Npe_i = Npe === nothing ? 0 : Int(Npe)
    predict = _build_predict(Npe_i ≥ 1, ti, tr, al, ae, Npe_i)

    T1_candidates = _t1_grid(T1_range, n_grid, T1_oracle, oracle_band)
    predict_cache = Vector{Vector{Float64}}(undef, n_grid)
    @inbounds for (i, T1) in enumerate(T1_candidates)
        predict_cache[i] = predict(T1)
    end

    scan = _scan_grid(predict_cache, T1_candidates, m; signed = signed)
    best_T1, best_A, best_sse, best_idx =
        scan.best_T1, scan.best_A, scan.best_sse, scan.best_idx

    T1_sigma = NaN
    if isfinite(best_T1) && best_T1 > 0 && isfinite(best_A)
        σ²_eff = _sigma_eff(best_sse, n, m, abs_noise_sigma, noise_sigma)
        T1_sigma = if sigma_method == :asymptotic
            _sigma_asymptotic(predict, best_T1, best_A, σ²_eff;
                              signed = signed)
        elseif sigma_method == :profile_likelihood
            _sigma_profile(scan.sse_grid, T1_candidates, best_sse, σ²_eff)
        else  # :bootstrap
            _sigma_bootstrap(predict_cache, T1_candidates, m,
                             best_idx, best_A;
                             signed = signed, n_bootstrap = n_bootstrap,
                             bootstrap_seed = bootstrap_seed)
        end
    end

    (T1 = best_T1, A = best_A,
     residual = sqrt(best_sse / n),
     T1_sigma = T1_sigma)
end

"""
    fit_t1_t2_generalized_ir(TIs, αs, TEs, mags; TRs, α_excs, Npe, T1_range,
                             T2_range, n_grid_t1, n_grid_t2, noise_sigma,
                             abs_noise_sigma, signed)

Joint `(T1, T2, A)` fit to `(TI, α_inv, TE, |S|)` tuples under the multiparametric
IR-spin-echo forward model

    |S| = |A · Mz_at_excite(T1; TI, TR, θ_inv, α_exc) · exp(−TE/T2)|

This is the IR-TSE / multi-echo analogue of `fit_t1_generalized_ir`. Where that
fit folds the (constant) `exp(−TE/T2)` echo attenuation into the amplitude `A`,
here `TE` varies per sample and `T2` is recovered alongside `T1` — so one IR-TSE
acquisition (TI varied across shots, TE varied across the echo train) yields both
a T1 and a T2 estimate.

Method: a 2-D log-grid scan over `(T1, T2)`. At each grid point the predicted
vector `Mz(T1; ·) .* exp(−TE/T2)` is formed (reusing the T1 forward model via
`_build_predict`) and the optimal `A` and SSE are closed-form — the same
`A = Σ m·|y| / Σ|y|²` ratio as the T1-only fit (`_A_and_sse`). Returns
`(T1, T2, A, residual, T1_sigma, T2_sigma)`.

`T1_sigma` / `T2_sigma` are **profile-likelihood** half-widths: the SSE grid is
marginalised onto each axis (min over the other parameter) and the span of
candidates within `σ²_eff` of the global minimum is reported (χ²₁, conf 0.683),
reusing `_sigma_profile`. `σ²_eff` uses the 3-parameter residual variance
`best_sse/(n−3)` for `n > 5`, else the supplied noise floor (`abs_noise_sigma²`
preferred). An unidentifiable parameter gets a wide σ — e.g. a constant TE leaves
T2 unconstrained and `T2_sigma` spans the whole grid.

Caller convention: as with `fit_t1_generalized_ir`, `mags` should already be
divided by `sin(α_exc)` per sample; do **not** pre-divide by `exp(−TE/T2)` — that
is what is being fit.
"""
function fit_t1_t2_generalized_ir(TIs::AbstractVector{<:Real},
                                  αs::AbstractVector{<:Real},
                                  TEs::AbstractVector{<:Real},
                                  mags::AbstractVector{<:Real};
                                  TRs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                                  α_excs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                                  Npe::Union{Nothing,Int} = nothing,
                                  T1_range::NTuple{2,<:Real} = (5e-3, 5.0),
                                  T2_range::NTuple{2,<:Real} = (1e-3, 3.0),
                                  n_grid_t1::Int = 200,
                                  n_grid_t2::Int = 120,
                                  noise_sigma::Union{Nothing,Real} = nothing,
                                  abs_noise_sigma::Union{Nothing,Real} = nothing,
                                  signed::Bool = false)
    n = length(TIs)
    n == length(αs) == length(TEs) == length(mags) ||
        error("TIs, αs, TEs, mags must be same length")
    n >= 2 || error("Need ≥2 samples for joint T1/T2 fit")
    TRs    === nothing || length(TRs)    == n || error("TRs length mismatch")
    α_excs === nothing || length(α_excs) == n || error("α_excs length mismatch")
    Npe    === nothing || Npe ≥ 1             || error("Npe must be ≥ 1")

    ti = Float64.(TIs)
    al = Float64.(αs)
    te = Float64.(TEs)
    m  = Float64.(mags)
    tr = TRs    === nothing ? fill(Inf, n) : Float64.(TRs)
    ae = α_excs === nothing ? fill(π/2, n) : Float64.(α_excs)

    Npe_i   = Npe === nothing ? 0 : Int(Npe)
    predict = _build_predict(Npe_i ≥ 1, ti, tr, al, ae, Npe_i)

    T1_candidates = exp.(range(log(T1_range[1]), log(T1_range[2]); length = n_grid_t1))
    T2_candidates = exp.(range(log(T2_range[1]), log(T2_range[2]); length = n_grid_t2))
    mz_cache    = [predict(T1) for T1 in T1_candidates]      # Mz over T1 grid
    decay_cache = [exp.(-te ./ T2) for T2 in T2_candidates]  # exp(−TE/T2) over T2 grid

    sse_grid = fill(Inf, n_grid_t1, n_grid_t2)
    best_sse = Inf; best_T1 = NaN; best_T2 = NaN; best_A = NaN
    tmp = Vector{Float64}(undef, n)
    @inbounds for i in 1:n_grid_t1
        mz = mz_cache[i]
        for j in 1:n_grid_t2
            dec = decay_cache[j]
            @. tmp = mz * dec                       # model prediction at (T1_i, T2_j)
            A, sse = _A_and_sse(tmp, m, signed)
            sse_grid[i, j] = sse
            if sse < best_sse
                best_sse = sse; best_A = A
                best_T1 = T1_candidates[i]; best_T2 = T2_candidates[j]
            end
        end
    end

    # σ²_eff with the 3-parameter (T1, T2, A) residual dof; n-gate at n > 5.
    σ²_floor = if abs_noise_sigma !== nothing
        Float64(abs_noise_sigma)^2
    elseif noise_sigma !== nothing
        (Float64(noise_sigma) * sqrt(sum(abs2, m) / n))^2
    else
        0.0
    end
    σ²_eff = n > 5 ? max(best_sse / (n - 3), σ²_floor) : σ²_floor

    T1_sigma = NaN; T2_sigma = NaN
    if isfinite(best_sse) && best_T1 > 0 && best_T2 > 0
        # Profile each parameter: minimise SSE over the other axis, then take the
        # χ²₁ span within σ²_eff of the minimum (reuses `_sigma_profile`).
        t1_profile = vec(minimum(sse_grid; dims = 2))
        t2_profile = vec(minimum(sse_grid; dims = 1))
        T1_sigma = _sigma_profile(t1_profile, T1_candidates, best_sse, σ²_eff)
        T2_sigma = _sigma_profile(t2_profile, T2_candidates, best_sse, σ²_eff)
    end

    (T1 = best_T1, T2 = best_T2, A = best_A,
     residual = sqrt(best_sse / n),
     T1_sigma = T1_sigma, T2_sigma = T2_sigma)
end
