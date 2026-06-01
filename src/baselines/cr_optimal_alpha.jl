# α-aware Cramér–Rao optimal schedule (ALPHA_DOF.md, reference point (c)) + Ernst-angle helpers
# (§3b). Kept SEPARATE from cr_optimal.jl so the α=90° path is byte-for-byte
# untouched and the two can be diffed.
#
# Modelling note — why the Fisher stays 2×2 (not 3×3):
#   The excitation flip angle α is a *design variable* (a commanded knob), not an
#   estimated parameter. `fit_t1_generalized_ir` takes α as a KNOWN input and
#   estimates only (T1, A). To keep the CR bound consistent with the estimator
#   the fleet actually uses, the Fisher information is over (T1, A) — a 2×2 matrix
#   — evaluated at each block's α. Optimising α means searching over where that
#   2×2 Jacobian is evaluated, not adding a third estimated parameter. At α=π/2
#   every function here reduces exactly to its cr_optimal.jl counterpart.

using LinearAlgebra: det, inv

# ── Ernst angle (§3b) ──────────────────────────────────────────────────────────

"""
    ernst_angle(TR, T1) -> α* (radians)

SNR-optimal flip angle for a spoiled steady-state acquisition,
`α* = acos(exp(−TR/T1))`. T1-dependent, so for a multi-T1 fleet it must be
evaluated at a representative T1 (we use the fleet median in
`ernst_fixed_schedule`).
"""
@inline ernst_angle(TR::Real, T1::Real) = acos(exp(-Float64(TR) / Float64(T1)))

"""
    ernst_fixed_schedule(TIs, TRs, T1_ref; α_bounds = (deg2rad(5), deg2rad(90))) -> αs

Per-block Ernst angle at the reference T1 (typically the fleet median), one α per
(TI, TR) block, clamped to `α_bounds`. Reuses the CR-opt `(TI, TR)` timing — only
the flip angle is set to the SNR-heuristic value.
"""
function ernst_fixed_schedule(TIs::AbstractVector, TRs::AbstractVector,
                              T1_ref::Real;
                              α_bounds::Tuple{<:Real,<:Real} = (deg2rad(5.0),
                                                                deg2rad(90.0)))
    @assert length(TIs) == length(TRs)
    αlo, αhi = Float64(α_bounds[1]), Float64(α_bounds[2])
    return [clamp(ernst_angle(TR, T1_ref), αlo, αhi) for TR in TRs]
end

# ── α-aware Jacobian / variance / objective (2×2 Fisher, α as design knob) ──────

"""
    jacobian_row_alpha(T1, A, TI, TR, α_exc; Npe, ε_T1 = 1e-4, ε_A = 1e-4)

∂S/∂T1 and ∂S/∂A at (T1, A) by central finite differences, with the signal model
evaluated at the given excitation flip angle `α_exc`. Same two columns as
`jacobian_row`; the only difference is the explicit per-block α.
"""
function jacobian_row_alpha(T1::Real, A::Real, TI::Real, TR::Real, α_exc::Real;
                            Npe::Int, ε_T1::Real = 1e-4, ε_A::Real = 1e-4)
    eT1 = max(ε_T1, ε_T1 * abs(T1))
    eA  = max(ε_A,  ε_A  * abs(A))
    dS_dT1 = (f_signal(T1 + eT1, A, TI, TR; Npe = Npe, α_exc = α_exc) -
              f_signal(T1 - eT1, A, TI, TR; Npe = Npe, α_exc = α_exc)) / (2 * eT1)
    dS_dA  = (f_signal(T1, A + eA, TI, TR; Npe = Npe, α_exc = α_exc) -
              f_signal(T1, A - eA, TI, TR; Npe = Npe, α_exc = α_exc)) / (2 * eA)
    return dS_dT1, dS_dA
end

"""
    cr_T1_variance_alpha(T1, TIs, TRs, αs; Npe, A = 1.0, σ_obs = 1.0)

Cramér–Rao lower bound on Var(T1*) for one sphere, given a schedule with per-block
flip angles `αs`. 2×2 Fisher over (T1, A); α only sets where the Jacobian is
evaluated. Reduces to `cr_T1_variance` when `αs .== π/2`.
"""
function cr_T1_variance_alpha(T1::Real, TIs::AbstractVector, TRs::AbstractVector,
                              αs::AbstractVector;
                              Npe::Int, A::Real = 1.0, σ_obs::Real = 1.0)
    n = length(TIs)
    @assert length(TRs) == n
    @assert length(αs) == n
    J = zeros(Float64, n, 2)
    @inbounds for k in 1:n
        dT1, dA = jacobian_row_alpha(T1, A, TIs[k], TRs[k], αs[k]; Npe = Npe)
        J[k, 1] = dT1
        J[k, 2] = dA
    end
    a = J[:, 1]' * J[:, 1]
    b = J[:, 1]' * J[:, 2]
    c = J[:, 2]' * J[:, 2]
    detF = a * c - b * b
    if detF ≤ 0 || !isfinite(detF)
        return Inf
    end
    var_T1 = σ_obs^2 * c / detF
    return (isfinite(var_T1) && var_T1 > 0) ? var_T1 : Inf
end

"""
    cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe, weights = nothing)

A-optimality `Σ_j w_j · Var(T1_j)/T1_j²` for the α-aware schedule.
"""
function cr_fleet_objective_alpha(T1s::AbstractVector, TIs::AbstractVector,
                                  TRs::AbstractVector, αs::AbstractVector;
                                  Npe::Int,
                                  weights::Union{Nothing, AbstractVector} = nothing)
    if weights === nothing
        weights = ones(length(T1s))
    end
    @assert length(weights) == length(T1s)
    L = 0.0
    @inbounds for j in eachindex(T1s)
        var_j = cr_T1_variance_alpha(T1s[j], TIs, TRs, αs; Npe = Npe)
        L += weights[j] * var_j / (T1s[j]^2)
    end
    return L
end

# ── α-aware schedule optimisation ───────────────────────────────────────────────

"""
    sample_random_schedule_alpha(rng, n_blocks; budget_s, Npe, ...) -> (TIs, TRs, αs) | nothing

Sample one budget-feasible (TI, TR, α) schedule. TI/TR sampled as in
`sample_random_schedule`; α sampled uniformly in `α_bounds`.
"""
function sample_random_schedule_alpha(rng::AbstractRNG, n_blocks::Int;
                                      budget_s::Real, Npe::Int,
                                      TI_lo::Real = 0.01, TI_hi::Real = 3.0,
                                      TR_lo_floor::Real = 0.5, TR_hi::Real = 5.0,
                                      α_lo::Real = deg2rad(5.0),
                                      α_hi::Real = deg2rad(90.0))
    for _ in 1:100
        TIs = exp.(log(TI_lo) .+ (log(TI_hi) - log(TI_lo)) .* rand(rng, n_blocks))
        TRs = similar(TIs)
        @inbounds for k in 1:n_blocks
            lo = max(TIs[k] + 0.05, TR_lo_floor)
            TRs[k] = exp(log(lo) + (log(TR_hi) - log(lo)) * rand(rng))
        end
        if schedule_time_s(TRs, Npe) ≤ budget_s
            αs = α_lo .+ (α_hi - α_lo) .* rand(rng, n_blocks)
            return TIs, TRs, αs
        end
    end
    return nothing
end

"""
    refine_coordinate_descent_alpha(TIs, TRs, αs, T1s; Npe, budget_s, ...)

Local refinement over (TI, TR, α) per block. TI/TR perturbed multiplicatively (as
in `refine_coordinate_descent`); α perturbed additively (clamped to `α_bounds`).
Accepts any budget-feasible move that lowers the objective.
"""
function refine_coordinate_descent_alpha(TIs::AbstractVector, TRs::AbstractVector,
                                         αs::AbstractVector, T1s::AbstractVector;
                                         Npe::Int, budget_s::Real,
                                         n_iter::Int = 50,
                                         step_factor::Real = 1.3,
                                         TI_lo::Real = 0.01, TI_hi::Real = 3.0,
                                         TR_lo_floor::Real = 0.5, TR_hi::Real = 5.0,
                                         α_lo::Real = deg2rad(5.0),
                                         α_hi::Real = deg2rad(90.0))
    TIs = collect(Float64, TIs)
    TRs = collect(Float64, TRs)
    αs  = collect(Float64, αs)
    L_best = cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe = Npe)
    factors  = [step_factor, 1/step_factor, 1.1, 1/1.1]
    α_steps  = deg2rad.([10.0, -10.0, 25.0, -25.0, 4.0, -4.0])

    for _ in 1:n_iter
        improved = false
        for k in 1:length(TIs)
            for f in factors
                # Perturb TI[k]
                new_TI = clamp(TIs[k] * f, TI_lo, TI_hi)
                new_TR_lo = max(new_TI + 0.05, TR_lo_floor)
                new_TR = clamp(TRs[k], new_TR_lo, TR_hi)
                old_TI, old_TR = TIs[k], TRs[k]
                TIs[k], TRs[k] = new_TI, new_TR
                if schedule_time_s(TRs, Npe) > budget_s
                    TIs[k], TRs[k] = old_TI, old_TR
                else
                    L = cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe = Npe)
                    if L < L_best
                        L_best = L; improved = true
                    else
                        TIs[k], TRs[k] = old_TI, old_TR
                    end
                end

                # Perturb TR[k]
                new_TR2 = clamp(TRs[k] * f, max(TIs[k] + 0.05, TR_lo_floor), TR_hi)
                old_TR2 = TRs[k]
                TRs[k] = new_TR2
                if schedule_time_s(TRs, Npe) > budget_s
                    TRs[k] = old_TR2
                else
                    L = cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe = Npe)
                    if L < L_best
                        L_best = L; improved = true
                    else
                        TRs[k] = old_TR2
                    end
                end
            end

            # Perturb α[k] (time-independent → no budget check needed)
            for da in α_steps
                new_α = clamp(αs[k] + da, α_lo, α_hi)
                new_α == αs[k] && continue
                old_α = αs[k]
                αs[k] = new_α
                L = cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe = Npe)
                if L < L_best
                    L_best = L; improved = true
                else
                    αs[k] = old_α
                end
            end
        end
        improved || break
    end
    return TIs, TRs, αs, L_best
end

"""
    cr_optimize_alpha(T1s; n_blocks, budget_s, Npe = 8, n_starts = 1000,
                       n_refine = 10, rng = MersenneTwister(0), α_bounds = ...)

Multi-start + coordinate-descent search for the α-aware CR-optimal schedule.
Returns `(TIs, TRs, αs, L)`.
"""
function cr_optimize_alpha(T1s::AbstractVector; n_blocks::Int, budget_s::Real,
                           Npe::Int = 8, n_starts::Int = 1000, n_refine::Int = 10,
                           rng::AbstractRNG = MersenneTwister(0),
                           α_bounds::Tuple{<:Real,<:Real} = (deg2rad(5.0),
                                                             deg2rad(90.0)))
    α_lo, α_hi = Float64(α_bounds[1]), Float64(α_bounds[2])
    candidates = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}[]
    for _ in 1:n_starts
        s = sample_random_schedule_alpha(rng, n_blocks; budget_s = budget_s,
                                         Npe = Npe, α_lo = α_lo, α_hi = α_hi)
        s === nothing && continue
        TIs, TRs, αs = s
        L = cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe = Npe)
        push!(candidates, (collect(TIs), collect(TRs), collect(αs), L))
    end
    isempty(candidates) &&
        return (TIs = Float64[], TRs = Float64[], αs = Float64[], L = Inf)
    sort!(candidates, by = c -> c[4])
    top = candidates[1:min(n_refine, length(candidates))]

    best_TIs, best_TRs, best_αs, best_L = top[1][1], top[1][2], top[1][3], top[1][4]
    for (TIs0, TRs0, αs0, _) in top
        TIs_r, TRs_r, αs_r, L_r = refine_coordinate_descent_alpha(
            TIs0, TRs0, αs0, T1s; Npe = Npe, budget_s = budget_s,
            α_lo = α_lo, α_hi = α_hi)
        if L_r < best_L
            best_L, best_TIs, best_TRs, best_αs = L_r, TIs_r, TRs_r, αs_r
        end
    end
    return (TIs = best_TIs, TRs = best_TRs, αs = best_αs, L = best_L)
end

"""
    cr_optimize_sweep_alpha(T1s; budget_s, Npe = 8,
                             n_block_grid = [4, 6, 8, 12, 16], kwargs...)

Sweep `n_blocks` for the α-aware solver; return the global optimum plus per-
n_blocks results. Mirrors `cr_optimize_sweep`.
"""
function cr_optimize_sweep_alpha(T1s::AbstractVector;
                                 budget_s::Real, Npe::Int = 8,
                                 n_block_grid::AbstractVector{Int} = [4, 6, 8, 12, 16],
                                 kwargs...)
    results = Dict{Int, NamedTuple}()
    best_n, best_L = 0, Inf
    for nb in n_block_grid
        block_time_s(0.5, Npe) * nb > budget_s && continue
        r = cr_optimize_alpha(T1s; n_blocks = nb, budget_s = budget_s, Npe = Npe,
                              kwargs...)
        isfinite(r.L) || continue
        results[nb] = r
        if r.L < best_L
            best_L, best_n = r.L, nb
        end
    end
    isempty(results) &&
        error("cr_optimize_sweep_alpha: no valid n_blocks fits budget=$budget_s s")
    return (n_blocks = best_n, schedule = results[best_n], all = results)
end
