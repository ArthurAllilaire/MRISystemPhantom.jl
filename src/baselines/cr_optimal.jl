# Cramér–Rao optimal fixed-block schedule for the F1+ forward model.
# E2_5_PLAN.md §4. Solves
#
#   argmin_{(TI_k, TR_k)_{k=1..n}}  Σ_j σ²_T1_j(schedule) / T1_j²
#   s.t.  Σ_k block_time(TR_k) ≤ budget_s
#          0.01 ≤ TI_k ≤ 3.0
#          max(TI_k + 0.05, 0.5) ≤ TR_k ≤ 5.0
#
# where σ²_T1_j = [(J_j^T J_j)⁻¹]_{T1, T1} from the F1+ forward model evaluated
# at the truth (T1_j, A_j = 1).
#
# Multi-start strategy: K random starts, score by objective, take top-N, refine
# each by coordinate descent. No external optimisation deps (avoids adding
# Optim.jl / ForwardDiff.jl to the project). Finite-difference Jacobian.

using Random
using Statistics: mean
using LinearAlgebra: I

"""
    block_time_s(TR, Npe; overhead_s = 0.05)

Approximate scan-time cost of one Npe-shot IR-SE block. Each shot ≈ TR seconds
(TI is consumed inside the TR window, then (TR − TI) recovery). Add a small
overhead for gradients/spoiling.
"""
@inline block_time_s(TR::Real, Npe::Int; overhead_s::Real = 0.05) =
    Npe * Float64(TR) + overhead_s

"""
    schedule_time_s(TRs, Npe)

Total scan time for a vector of TR values, Npe shots each.
"""
schedule_time_s(TRs::AbstractVector, Npe::Int; kw...) =
    sum(block_time_s(TR, Npe; kw...) for TR in TRs)

# ── F1+ forward and its Jacobian (finite-difference) ──────────────────────────

"""
    f_signal(T1, A, TI, TR; Npe, θ_inv = π, α_exc = π/2)

|S(T1, A; TI, TR)| under F1+. Wraps `transient_mz_at_excite_npe`.
"""
@inline function f_signal(T1::Real, A::Real, TI::Real, TR::Real;
                           Npe::Int, θ_inv::Real = π, α_exc::Real = π/2)
    A * abs(transient_mz_at_excite_npe(Float64(T1), Float64(TI), Float64(TR),
                                        Float64(θ_inv), Float64(α_exc); Npe = Npe))
end

"""
    jacobian_row(T1, A, TI, TR; Npe, ε_T1 = 1e-4, ε_A = 1e-4)

Returns ∂S/∂T1 and ∂S/∂A at (T1, A) by central finite differences.
"""
function jacobian_row(T1::Real, A::Real, TI::Real, TR::Real;
                       Npe::Int, ε_T1::Real = 1e-4, ε_A::Real = 1e-4)
    # ε_T1 in absolute units; for very small T1 (e.g. 0.02 s), use a relative step
    eT1 = max(ε_T1, ε_T1 * abs(T1))
    eA  = max(ε_A,  ε_A  * abs(A))
    dS_dT1 = (f_signal(T1 + eT1, A, TI, TR; Npe = Npe) -
              f_signal(T1 - eT1, A, TI, TR; Npe = Npe)) / (2 * eT1)
    dS_dA  = (f_signal(T1, A + eA, TI, TR; Npe = Npe) -
              f_signal(T1, A - eA, TI, TR; Npe = Npe)) / (2 * eA)
    return dS_dT1, dS_dA
end

# ── CR objective ──────────────────────────────────────────────────────────────

"""
    cr_T1_variance(T1, TIs, TRs; Npe, A = 1.0, σ_obs = 1.0)

Cramér–Rao lower bound on Var(T1*) for one sphere with true T1, given the
schedule (TIs, TRs). σ_obs is the per-measurement noise σ; σ_T1 ∝ σ_obs, and we
use σ_obs = 1 here so the returned variance is in *units of σ_obs²*. The fleet
objective normalises by T1², so absolute σ_obs cancels — only the schedule's
information geometry matters.
"""
function cr_T1_variance(T1::Real, TIs::AbstractVector, TRs::AbstractVector;
                         Npe::Int, A::Real = 1.0, σ_obs::Real = 1.0)
    n = length(TIs)
    @assert length(TRs) == n
    # Build Jacobian J ∈ R^{n × 2} at (T1, A)
    J = zeros(Float64, n, 2)
    @inbounds for k in 1:n
        dT1, dA = jacobian_row(T1, A, TIs[k], TRs[k]; Npe = Npe)
        J[k, 1] = dT1
        J[k, 2] = dA
    end
    # (JᵀJ)⁻¹ — 2×2 closed form
    a = J[:, 1]' * J[:, 1]
    b = J[:, 1]' * J[:, 2]
    c = J[:, 2]' * J[:, 2]
    det = a * c - b * b
    if det ≤ 0 || !isfinite(det)
        return Inf      # information matrix singular → T1 unidentified
    end
    var_T1 = σ_obs^2 * c / det
    return var_T1
end

"""
    cr_fleet_objective(T1s, TIs, TRs; Npe, weights = nothing)

Weighted sum of σ²_T1_j / T1_j² across the fleet. By default each sphere is
weighted equally; pass `weights` to override (e.g. for a per-decade weighted
objective).
"""
function cr_fleet_objective(T1s::AbstractVector,
                             TIs::AbstractVector, TRs::AbstractVector;
                             Npe::Int,
                             weights::Union{Nothing, AbstractVector} = nothing)
    if weights === nothing
        weights = ones(length(T1s))
    end
    @assert length(weights) == length(T1s)
    L = 0.0
    @inbounds for j in eachindex(T1s)
        var_j = cr_T1_variance(T1s[j], TIs, TRs; Npe = Npe)
        L += weights[j] * var_j / (T1s[j]^2)
    end
    return L
end

# ── Schedule optimisation ─────────────────────────────────────────────────────

"""
    sample_random_schedule(rng, n_blocks; budget_s, Npe,
                            TI_lo = 0.01, TI_hi = 3.0,
                            TR_lo_floor = 0.5, TR_hi = 5.0)

Sample one random schedule that respects the budget. Re-samples up to 100 times
if the first draw exceeds budget. Returns `nothing` if no valid sample found.
"""
function sample_random_schedule(rng::AbstractRNG, n_blocks::Int;
                                  budget_s::Real, Npe::Int,
                                  TI_lo::Real = 0.01, TI_hi::Real = 3.0,
                                  TR_lo_floor::Real = 0.5, TR_hi::Real = 5.0)
    for _ in 1:100
        TIs = exp.(log(TI_lo) .+ (log(TI_hi) - log(TI_lo)) .* rand(rng, n_blocks))
        TRs = similar(TIs)
        @inbounds for k in 1:n_blocks
            lo = max(TIs[k] + 0.05, TR_lo_floor)
            TRs[k] = exp(log(lo) + (log(TR_hi) - log(lo)) * rand(rng))
        end
        if schedule_time_s(TRs, Npe) ≤ budget_s
            return TIs, TRs
        end
    end
    return nothing
end

"""
    refine_coordinate_descent(TIs, TRs, T1s; Npe, budget_s, n_iter = 50,
                                step_factor = 1.3)

Local refinement: for each block, try multiplying TI and TR by `step_factor`
and `1/step_factor` (and a smaller perturbation). Accept any move that lowers
the objective and keeps total time ≤ budget. Repeats `n_iter` passes.
"""
function refine_coordinate_descent(TIs::AbstractVector, TRs::AbstractVector,
                                     T1s::AbstractVector;
                                     Npe::Int, budget_s::Real,
                                     n_iter::Int = 50,
                                     step_factor::Real = 1.3,
                                     TI_lo::Real = 0.01, TI_hi::Real = 3.0,
                                     TR_lo_floor::Real = 0.5, TR_hi::Real = 5.0)
    TIs = collect(Float64, TIs)
    TRs = collect(Float64, TRs)
    L_best = cr_fleet_objective(T1s, TIs, TRs; Npe = Npe)
    factors = [step_factor, 1/step_factor, 1.1, 1/1.1]

    for iter in 1:n_iter
        improved = false
        for k in 1:length(TIs)
            for f in factors
                # Try perturbing TI[k]
                new_TI = clamp(TIs[k] * f, TI_lo, TI_hi)
                new_TR_lo = max(new_TI + 0.05, TR_lo_floor)
                new_TR = clamp(TRs[k], new_TR_lo, TR_hi)
                # Save and apply
                old_TI, old_TR = TIs[k], TRs[k]
                TIs[k], TRs[k] = new_TI, new_TR
                if schedule_time_s(TRs, Npe) > budget_s
                    TIs[k], TRs[k] = old_TI, old_TR
                    continue
                end
                L = cr_fleet_objective(T1s, TIs, TRs; Npe = Npe)
                if L < L_best
                    L_best = L
                    improved = true
                else
                    TIs[k], TRs[k] = old_TI, old_TR
                end

                # Try perturbing TR[k]
                new_TR2 = clamp(TRs[k] * f, max(TIs[k] + 0.05, TR_lo_floor), TR_hi)
                old_TR2 = TRs[k]
                TRs[k] = new_TR2
                if schedule_time_s(TRs, Npe) > budget_s
                    TRs[k] = old_TR2
                    continue
                end
                L = cr_fleet_objective(T1s, TIs, TRs; Npe = Npe)
                if L < L_best
                    L_best = L
                    improved = true
                else
                    TRs[k] = old_TR2
                end
            end
        end
        improved || break
    end
    return TIs, TRs, L_best
end

"""
    cr_optimize(T1s; n_blocks, budget_s, Npe = 8,
                  n_starts = 1000, n_refine = 10, rng = MersenneTwister(0))

Multi-start + coordinate-descent search for the CR-optimal schedule.
Returns NamedTuple `(TIs, TRs, L)` of the best schedule found.
"""
function cr_optimize(T1s::AbstractVector; n_blocks::Int, budget_s::Real,
                       Npe::Int = 8, n_starts::Int = 1000, n_refine::Int = 10,
                       rng::AbstractRNG = MersenneTwister(0))
    # Phase 1: random sampling
    candidates = Tuple{Vector{Float64}, Vector{Float64}, Float64}[]
    for _ in 1:n_starts
        s = sample_random_schedule(rng, n_blocks; budget_s = budget_s, Npe = Npe)
        s === nothing && continue
        TIs, TRs = s
        L = cr_fleet_objective(T1s, TIs, TRs; Npe = Npe)
        push!(candidates, (collect(TIs), collect(TRs), L))
    end
    # No valid samples (budget too tight for this n_blocks) → bail
    isempty(candidates) && return (TIs = Float64[], TRs = Float64[], L = Inf)
    # Sort by objective, take top n_refine
    sort!(candidates, by = c -> c[3])
    top = candidates[1:min(n_refine, length(candidates))]

    # Phase 2: refine each
    best_TIs = top[1][1]
    best_TRs = top[1][2]
    best_L   = top[1][3]
    for (TIs0, TRs0, _) in top
        TIs_r, TRs_r, L_r = refine_coordinate_descent(
            TIs0, TRs0, T1s; Npe = Npe, budget_s = budget_s)
        if L_r < best_L
            best_L = L_r
            best_TIs = TIs_r
            best_TRs = TRs_r
        end
    end
    return (TIs = best_TIs, TRs = best_TRs, L = best_L)
end

"""
    cr_optimize_sweep(T1s; budget_s, Npe = 8, n_block_grid = [4, 6, 8, 12, 16],
                        kwargs...)

Sweep `n_blocks` and return the global optimum across the sweep, plus per-
n_blocks results for diagnostics.
"""
function cr_optimize_sweep(T1s::AbstractVector;
                             budget_s::Real, Npe::Int = 8,
                             n_block_grid::AbstractVector{Int} = [4, 6, 8, 12, 16],
                             kwargs...)
    results = Dict{Int, NamedTuple}()
    best_n, best_L = 0, Inf
    for nb in n_block_grid
        # Skip if budget can't fit even one block at TR_lo_floor
        block_time_s(0.5, Npe) * nb > budget_s && continue
        r = cr_optimize(T1s; n_blocks = nb, budget_s = budget_s, Npe = Npe,
                          kwargs...)
        # Skip if no valid schedule was sampled (e.g. nb too large for budget)
        isfinite(r.L) || continue
        results[nb] = r
        if r.L < best_L
            best_L = r.L
            best_n = nb
        end
    end
    isempty(results) && error("cr_optimize_sweep: no valid n_blocks fits budget=$budget_s s")
    return (n_blocks = best_n, schedule = results[best_n], all = results)
end
