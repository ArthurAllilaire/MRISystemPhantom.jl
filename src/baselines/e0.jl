# E0 — conventional-sequence baseline (PLAN.md §4 E0).
# Runs IR on each T1-array sphere, multi-TE SE on each T2-array sphere,
# fits T1 / T2 monoexponentials, and compares to the manual values.
# The fits become the yardstick the RL agent has to match (on accuracy)
# and beat (on scan time).

"""
    measure_ir_signal(; T1, T2, TI, amp_T = 2e-6, n_adc = 16, dur_adc = 2e-3)

Run a single IR simulation on a single-spin phantom with the given
relaxation times and return the magnitude of the first ADC sample.
"""
function measure_ir_signal(; T1::Real, T2::Real, TI::Real,
                             amp_T::Real = 2e-6,
                             n_adc::Int  = 16,
                             dur_adc::Real = 2e-3,
                             scanner::Scanner = Scanner())
    obj = single_spin_phantom(T1 = T1, T2 = T2)
    seq = ir_sequence(TI; amp_T = amp_T, n_adc = n_adc, dur_adc = dur_adc)
    raw = Suppressor.@suppress simulate(obj, seq, scanner)
    abs(raw.profiles[1].data[1, 1])
end

"""
    measure_se_signal(; T1, T2, TE, amp_T = 2e-6, n_adc = 32, dur_adc = 2e-3)

Run a single SE simulation on a single-spin phantom and return the
magnitude of the sample nearest the echo centre.
"""
function measure_se_signal(; T1::Real, T2::Real, TE::Real,
                             amp_T::Real = 20e-6,
                             n_adc::Int  = 32,
                             dur_adc::Real = min(2e-3, TE/4),
                             scanner::Scanner = Scanner())
    obj = single_spin_phantom(T1 = T1, T2 = T2)
    seq = se_sequence(TE; amp_T = amp_T, n_adc = n_adc, dur_adc = dur_adc)
    raw = Suppressor.@suppress simulate(obj, seq, scanner)
    samples = raw.profiles[1].data[:, 1]
    abs(samples[n_adc ÷ 2 + 1])         # DC index for even convention (N/2+1)
end

"""
    measure_mse_signal(; T1, T2, ESP, n_echoes, amp_T = 20e-6, n_adc = 1, ...)

Run a single CPMG (`mse_sequence`) simulation on a single-spin phantom and
return the vector of `n_echoes` echo magnitudes — the centre sample of each
echo's ADC block. Echo `k` sits at `TE = k·ESP`, so the result tracks
`exp(−k·ESP/T2)`: the whole T2 decay curve from **one** acquisition.
"""
function measure_mse_signal(; T1::Real, T2::Real, ESP::Real, n_echoes::Int,
                              amp_T::Real = 20e-6,
                              n_adc::Int  = 1,
                              dur_adc::Real = min(1e-3, ESP/4),
                              scanner::Scanner = Scanner())
    obj = single_spin_phantom(T1 = T1, T2 = T2)
    seq = mse_sequence(ESP, n_echoes; amp_T = amp_T, n_adc = n_adc,
                       dur_adc = dur_adc)
    raw = Suppressor.@suppress simulate(obj, seq, scanner)
    # One ADC block (profile) per echo; centre sample of each.
    [abs(raw.profiles[k].data[n_adc ÷ 2 + 1, 1]) for k in 1:n_echoes]
end

"""
    adaptive_TI_schedule(T1_hint; n = 10, lo_factor = 1/20, hi_factor = 5)

Log-spaced TI values tailored to a sphere with approximate T1 `T1_hint`.
Spans T1_hint/20 to 5·T1_hint — wide enough for a 3-parameter IR fit
(needs samples both through the null time and on the plateau).
"""
function adaptive_TI_schedule(T1_hint::Real; n::Int = 10,
                              lo_factor::Real = 1/20, hi_factor::Real = 5)
    lo = max(1e-3, lo_factor * T1_hint)
    hi = hi_factor * T1_hint
    exp.(range(log(lo), log(hi); length = n))
end

"""
    adaptive_TE_schedule(T2_hint; n = 10, lo_factor = 1/15, hi_factor = 3)

Log-spaced TE values tailored to a sphere with approximate T2 `T2_hint`.
Spans T2_hint/15 to 3·T2_hint (past ~3·T2 the signal is below noise).
Clamped from below at 1 ms so the default SE pulses fit.
"""
function adaptive_TE_schedule(T2_hint::Real; n::Int = 10,
                              lo_factor::Real = 1/15, hi_factor::Real = 3)
    lo = max(1e-3, lo_factor * T2_hint)
    hi = hi_factor * T2_hint
    exp.(range(log(lo), log(hi); length = n))
end

# Non-adaptive schedules left in for callers who want a fixed grid.
default_TI_schedule(; n::Int = 10) =
    exp.(range(log(5e-3), log(3.0); length = n))
default_TE_schedule(; n::Int = 10) =
    exp.(range(log(3e-3), log(1.5); length = n))

"""
    measure_t1(T1_true, T2_true; TIs, T1_hint, kwargs...)

Run an IR sweep and fit T1. If `TIs` is not provided, builds one from
`T1_hint` (defaults to `T1_true`) via `adaptive_TI_schedule`.
"""
function measure_t1(T1_true::Real, T2_true::Real;
                    T1_hint::Real = T1_true,
                    TIs::AbstractVector{<:Real} = adaptive_TI_schedule(T1_hint),
                    kwargs...)
    mags = [measure_ir_signal(; T1 = T1_true, T2 = T2_true, TI = TI,
                               kwargs...) for TI in TIs]
    # grid T1_range should bracket the hint; use a generous window.
    fit = fit_t1_ir(TIs, mags;
                    T1_range = (minimum(TIs) / 2, maximum(TIs) * 2))
    (T1_est = fit.T1, TIs = TIs, mags = mags, fit = fit)
end

"""
    measure_t2(T1_true, T2_true; mode = :multishot, TEs, T2_hint, kwargs...)

Run a T2 sweep and fit T2. Two acquisition modes:

- `:multishot` (default, conventional): one single-echo SE per TE in `TEs`
  (`length(TEs)` separate TR-long shots). If `TEs` is not provided, builds one
  from `T2_hint` (defaults to `T2_true`) via `adaptive_TE_schedule`.
- `:cpmg`: a single multi-echo (`mse_sequence`) acquisition with `n_echoes`
  echoes spaced `ESP` apart — the whole decay curve in one shot. Defaults derive
  `ESP`/`n_echoes` from `T2_hint` so the train spans ~3·T2.

Both fit the same mono-exponential `fit_t2_se` over the per-echo times. Returns
`(T2_est, TEs, mags, fit)`; `:cpmg` additionally exposes `ESP`/`n_echoes`.
"""
function measure_t2(T1_true::Real, T2_true::Real;
                    mode::Symbol = :multishot,
                    T2_hint::Real = T2_true,
                    TEs::AbstractVector{<:Real} = adaptive_TE_schedule(T2_hint),
                    n_echoes::Int = 12,
                    ESP::Real = max(3e-3, 3 * T2_hint / n_echoes),
                    kwargs...)
    if mode === :cpmg
        TEs_cpmg = ESP .* (1:n_echoes)
        mags = measure_mse_signal(; T1 = T1_true, T2 = T2_true,
                                   ESP = ESP, n_echoes = n_echoes, kwargs...)
        fit = fit_t2_se(collect(TEs_cpmg), mags)
        return (T2_est = fit.T2, TEs = collect(TEs_cpmg), mags = mags,
                fit = fit, ESP = ESP, n_echoes = n_echoes)
    elseif mode === :multishot
        mags = [measure_se_signal(; T1 = T1_true, T2 = T2_true, TE = TE,
                                   kwargs...) for TE in TEs]
        fit = fit_t2_se(TEs, mags)
        return (T2_est = fit.T2, TEs = TEs, mags = mags, fit = fit)
    else
        error("measure_t2 mode must be :multishot or :cpmg; got :$mode")
    end
end

"""
    run_e0(; field = :T3, verbose = true)

Full E0 baseline: estimate T1 for every T1-array sphere and T2 for every
T2-array sphere at the given field strength. Returns a NamedTuple with
per-sphere truth, estimate, and percentage error plus overall MAPEs.
"""
function run_e0(; field::Symbol = :T3, verbose::Bool = true,
                 n_TIs::Int = 10, n_TEs::Int = 10)
    t1_true  = T1_ARRAY[field]
    t2_of_t1 = T2_OF_T1_ARRAY[field]
    t2_true  = T2_ARRAY[field]
    sys = scanner_for_field(field)

    verbose && @info "E0: T1 mapping on the T1 array" field n_TIs
    t1_est = zeros(14)
    for i in 1:14
        TIs = adaptive_TI_schedule(t1_true[i]; n = n_TIs)
        res = measure_t1(t1_true[i], t2_of_t1[i]; TIs = TIs, scanner = sys)
        t1_est[i] = res.T1_est
        verbose && @info "  T1-$i" T1_true_ms = 1000*t1_true[i] T1_est_ms = 1000*res.T1_est
    end

    verbose && @info "E0: T2 mapping on the T2 array" field n_TEs
    t2_est = zeros(14)
    t1_of_t2 = T1_OF_T2_ARRAY[field]
    for i in 1:14
        TEs = adaptive_TE_schedule(t2_true[i]; n = n_TEs)
        res = measure_t2(t1_of_t2[i], t2_true[i]; TEs = TEs, scanner = sys)
        t2_est[i] = res.T2_est
        verbose && @info "  T2-$i" T2_true_ms = 1000*t2_true[i] T2_est_ms = 1000*res.T2_est
    end

    t1_err_pct = 100 .* abs.(t1_est .- t1_true) ./ t1_true
    t2_err_pct = 100 .* abs.(t2_est .- t2_true) ./ t2_true
    t1_mape = sum(t1_err_pct) / length(t1_err_pct)
    t2_mape = sum(t2_err_pct) / length(t2_err_pct)

    if verbose
        @info "E0 summary" field T1_MAPE_pct = t1_mape T2_MAPE_pct = t2_mape
    end

    (field   = field,
     T1_true = t1_true, T1_est = t1_est,
     T1_err_pct = t1_err_pct, T1_MAPE_pct = t1_mape,
     T2_true = t2_true, T2_est = t2_est,
     T2_err_pct = t2_err_pct, T2_MAPE_pct = t2_mape)
end
