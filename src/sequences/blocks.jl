# Pulseq-style sequence building blocks used by E0 and the later RL
# experiments. Blocks are parameterised functions returning a `Sequence`
# that can be concatenated with `+=` or simulated directly.
#
# RF timing convention matches `01-FID.jl`:
#   α = 2π · γ · B1 · duration
# where γ is in Hz/T (42.58 MHz/T for protons) and B1 in T.

"""
    rf_duration(α; amp_T)

Duration (s) of a rectangular RF pulse that produces flip angle `α`
(radians) at B1 = `amp_T` tesla, assuming on-resonance.
"""
rf_duration(α; amp_T = 2e-6) = α / (2π * γ * amp_T)

"""
    ir_sequence(TI; amp_T, n_adc, dur_adc)

Inversion-recovery sequence: 180° inversion → delay TI → 90° excitation →
ADC. Returns a `Sequence`. The first ADC sample is the FID amplitude at
the start of readout and, for `amp_T` small enough that T2 decay during
the pulses is negligible, tracks `|1 − 2·exp(−TI/T1)|` for a
Mz-equilibrium spin.
"""
function ir_sequence(TI::Real;
                     amp_T::Real = 2e-6,
                     n_adc::Int  = 16,
                     dur_adc::Real = 2e-3,
                     adc_delay::Real = 0.0)
    d90  = rf_duration(π/2;  amp_T = amp_T)
    d180 = rf_duration(π;    amp_T = amp_T)
    seq  = Sequence()
    seq += RF(amp_T, d180)
    seq += Delay(TI)
    seq += RF(amp_T, d90)
    seq += ADC(n_adc, dur_adc, adc_delay)
    seq
end

"""
    se_sequence(TE; amp_T, n_adc, dur_adc)

Spin-echo sequence: 90° → delay TE/2 → 180° → delay TE/2 → ADC centred
near the echo. The middle ADC sample approximates the echo peak, whose
magnitude tracks `S0·exp(−TE/T2)`.
"""
function se_sequence(TE::Real;
                     amp_T::Real = 20e-6,    # harder pulses → short d180
                     n_adc::Int  = 32,       # even convention: -N/2 to N/2-1, DC at index N/2+1
                     dur_adc::Real = min(2e-3, TE/4))
    d90  = rf_duration(π/2; amp_T = amp_T)
    d180 = rf_duration(π;   amp_T = amp_T)

    # Center ADC on the echo: between the end of the 180° and the ADC
    # midpoint we want TE/2 − d180/2 (from pulse centre to echo) minus
    # half the ADC window.
    pre  = TE/2 - d90/2  - d180/2
    post = TE/2 - d180/2 - dur_adc/2
    pre  <= 0 && error("TE too short for 90° pulse duration; got TE=$TE")
    post <= 0 && error("TE too short for ADC window; got TE=$TE")

    seq  = Sequence()
    seq += RF(amp_T, d90)
    seq += Delay(pre)
    seq += RF(amp_T, d180)
    seq += Delay(post)
    seq += ADC(n_adc, dur_adc, 0.0)
    seq
end

"""
    mse_sequence(ESP, n_echoes; amp_T, n_adc, dur_adc)

Multi-echo spin echo (CPMG): a single 90° excitation followed by a train of
`n_echoes` 180° refocusing pulses, each with an ADC centred on its echo:

    90° → [ESP/2 → 180° → ESP/2 → ADC]·n_echoes

Echo `k` forms at `t = k·ESP` after the excitation, so the centre ADC sample
of block `k` tracks `S0·exp(−k·ESP/T2)` for a Mz-equilibrium spin under perfect
refocusing + transverse spoiling (the analytic `mse_signal`). This samples the
whole T2 decay curve in **one TR**, versus one TE per shot for `se_sequence`.

`n_echoes = 1` reproduces `se_sequence(ESP)` (one echo at TE = ESP). Errors if
`ESP` is too short for the pulse/ADC durations. Imperfect 180° pulses introduce
stimulated/indirect echoes that break the mono-exponential model (would need
EPG); the digital twin uses exact 180°, so mono-exp holds — and exact 180°s mean
no crusher pair is needed to keep the echo clean, so this block carries none
(adding gradient crushers would have to be budgeted into the ESP delays).
"""
function mse_sequence(ESP::Real, n_echoes::Int;
                      amp_T::Real   = 20e-6,
                      n_adc::Int    = 1,
                      dur_adc::Real = min(1e-3, ESP/4))
    n_echoes >= 1 || error("n_echoes must be ≥ 1; got $n_echoes")
    d90  = rf_duration(π/2; amp_T = amp_T)
    d180 = rf_duration(π;   amp_T = amp_T)

    # Centre-to-centre spacing is ESP between successive echoes (and the first
    # 180° sits ESP/2 after the 90°). Convert to inter-block delays.
    gap_first = ESP/2 - d90/2  - d180/2      # 90° centre → first 180° centre
    gap_half  = ESP/2 - d180/2 - dur_adc/2   # 180° centre → echo, and echo → next 180°
    gap_first <= 0 && error("ESP too short for 90°/180° pulse durations; got ESP=$ESP")
    gap_half  <= 0 && error("ESP too short for 180° pulse + ADC window; got ESP=$ESP")

    seq = Sequence()
    seq += RF(amp_T, d90)
    for k in 1:n_echoes
        seq += Delay(k == 1 ? gap_first : gap_half)
        seq += RF(amp_T, d180)
        seq += Delay(gap_half)
        seq += ADC(n_adc, dur_adc, 0.0)
    end
    seq
end

"""
    generalized_ir_signal(T1, T2; TI, α, n_adc = 64, dur_adc = 2e-3)

Analytical (closed-form) magnitude readout of an IR-prep / 90°-excite /
ADC block. The prep pulse tips Mz by angle `α` (so α = π is canonical
inversion recovery, α = π/2 is saturation recovery, α = 10° is a
small-tip prep). Assumes perfect transverse spoiling after the prep so
that only Mz is carried forward to the excitation. The returned vector is

    |Mz(TI)| · exp(−t_i / T2),  t_i ∈ [0, dur_adc]

with `n_adc` uniformly spaced samples. Orders of magnitude faster than
`simulate()` and equivalent in the single-spin limit — exactly what
PLAN.md §7 calls for on the training hot path.
"""
function generalized_ir_signal(T1::Real, T2::Real;
                               TI::Real, α::Real,
                               n_adc::Int = 64,
                               dur_adc::Real = 2e-3)
    Mz_after_prep = cos(α)
    Mz_at_excite  = 1 - (1 - Mz_after_prep) * exp(-TI / T1)
    amp           = abs(Mz_at_excite)
    ts            = range(0, dur_adc; length = n_adc)
    Float64[amp * exp(-t / T2) for t in ts]
end

"""
    mse_signal(T2; ESP, n_echoes)

Analytical (closed-form) echo-train magnitudes for the CPMG `mse_sequence`.
Under perfect 90°/180° pulses and transverse spoiling, echo `k` (at `t = k·ESP`
after the excitation) has magnitude `exp(−k·ESP/T2)` for a unit-amplitude spin:

    [exp(−k·ESP/T2)  for k in 1:n_echoes]

So `mse_signal(T2; ESP, n) ./` an overall amplitude `A` is exactly what
`fit_t2_se(ESP .* (1:n), mags)` inverts. Orders of magnitude faster than
`simulate()` and equivalent in the single-spin limit — the training hot path and
the test oracle, mirroring `generalized_ir_signal` for T1.
"""
function mse_signal(T2::Real; ESP::Real, n_echoes::Int)
    n_echoes >= 1 || error("n_echoes must be ≥ 1; got $n_echoes")
    Float64[exp(-k * Float64(ESP) / T2) for k in 1:n_echoes]
end

"""
    SpoilerConfig(; enabled=false, amp_T=30e-3, dur=5e-3, axis=:z)

Bundles spoiler/crusher parameters so sequence builders can share one
opinion of "what a spoiler is". `axis` selects which gradient axis carries
the spoil moment (`:x`, `:y`, or `:z`).
"""
Base.@kwdef struct SpoilerConfig
    enabled::Bool   = false
    amp_T::Float64  = 30e-3
    dur::Float64    = 5e-3
    axis::Symbol    = :z
end

"""
    apply_spoiler(seq, cfg::SpoilerConfig) → Sequence

Append a gradient-only spoiler block on `cfg.axis` when `cfg.enabled`,
otherwise return `seq` unchanged. Mutates `seq` and returns it for
chaining inside `@addblocks` loops.
"""
function apply_spoiler(seq, cfg::SpoilerConfig)
    cfg.enabled || return seq
    g = Grad(cfg.amp_T, cfg.dur)
    cfg.axis === :z && return addblock!(seq; z=g)
    cfg.axis === :y && return addblock!(seq; y=g)
    cfg.axis === :x && return addblock!(seq; x=g)
    error("unknown spoiler axis $(cfg.axis)")
end

"""
    ir_se_2d_sequence(TI, TE, TR; α_exc, FOV, Nfe, Npe, amp_T, spoiler) → Sequence

Multi-shot 2D spin-echo IR sequence. `Npe` shots are concatenated, each with a
different Gy phase-encode amplitude. Returns a `Sequence` with exactly `Npe`
ADC blocks (one `Nfe`-sample readout per phase-encode step).

Gradient convention (spin-echo k-space):
  - Positive Gx prewinder after excitation: kx → +kmax_x
  - 180° refocus conjugates phase: kx_eff → −kmax_x
  - Positive Gx readout drives kx_eff from −kmax_x to +kmax_x; echo at midpoint

After `simulate()`, reconstruct the magnitude image as:
  `abs.(ifft(ksp, (1,2)))` where `ksp[k,:] = raw.profiles[k].data[:,1]`

Sphere centre (x,y) [m] → image pixel (i_pe, i_fe):
  `i_fe = mod(round(Int, x*Nfe/FOV), Nfe) + 1`
  `i_pe = mod(round(Int, y*Npe/FOV), Npe) + 1`

A spin echo is used to remove T2* decay (much faster and less predictable
than T2). Pass a non-default `spoiler::SpoilerConfig` to add Gz crushers
around the refocus pulse plus a TR spoiler — required at short TR to
prevent FID contamination and residual Mxy carryover between shots.
"""
function ir_se_2d_sequence(TI::Real, TE::Real, TR::Real;
                            α_exc::Real             = π / 2,
                            FOV::Real               = 0.2,
                            Nfe::Int                = 16,
                            Npe::Int                = 8,
                            amp_T::Real             = 20e-6,
                            spoiler::SpoilerConfig  = SpoilerConfig())
    # KomaMRI convention: k = γ * ∫G dt with γ in Hz/T (no 2π factor — see
    # `KomaMRIBase.jl:17` const global γ = 42.5774688e6 Hz/T). Previous code
    # divided by γ_rad = 2π·γ_Hz and produced gradients 2π× too small,
    # collapsing the effective image FOV. Fixed by using γ_Hz directly.
    γ_Hz = 42.577e6   # proton gyromagnetic ratio [Hz/T], Koma convention

    d_inv = rf_duration(π;     amp_T = amp_T)
    d_exc = rf_duration(α_exc; amp_T = amp_T)
    d_ref = rf_duration(π;     amp_T = amp_T)

    dur_adc = 1e-3            # readout window [s]
    dur_pe  = dur_adc / 2     # prewinder / phase-encode duration [s]

    kmax_x = Nfe / (2.0 * FOV)
    Gx_pre = kmax_x / (γ_Hz * dur_pe)
    Gx_ro  = 2.0 * kmax_x / (γ_Hz * dur_adc)

    # Phase-encode steps: even convention, kmin = -Npe/2, kmax = Npe/2-1.
    # DC (ky=0) lands at k = Npe÷2+1, matching the -N/2…N/2-1 FFT layout.
    Δky      = 1.0 / FOV
    ky_steps = [(k - 1 - Npe ÷ 2) * Δky for k in 1:Npe]

    # Spoiler duration absorbed into TE budget when enabled.
    d_crush = spoiler.enabled ? spoiler.dur : 0.0

    ti_d  = max(TI  - d_inv / 2 - d_exc / 2, 0.0)
    te1_d = max(TE / 2 - d_exc / 2 - dur_pe - d_crush - d_ref / 2, 0.0)
    te2_d = max(TE / 2 - d_ref / 2 - d_crush - dur_adc / 2, 0.0)

    shot_time = d_inv + ti_d + d_exc + dur_pe + te1_d +
                d_crush + d_ref + d_crush + te2_d + dur_adc + d_crush
    tr_d      = max(TR - shot_time, 0.0)

    seq = Sequence()
    @addblocks for k in 1:Npe
        # Negative prewinder: after the 180° conjugates phase, the readout
        # samples ky = +ky_steps[k] with no Gy during ADC (clean Cartesian).
        Gy_k = -ky_steps[k] / (γ_Hz * dur_pe)

        seq += (RF(amp_T, d_inv),)                                          # 1. 180° inversion
        ti_d > 1e-9 && (seq += Delay(ti_d))                                 # 2. TI
        seq += (RF(amp_T, d_exc),)                                          # 3. excitation
        seq += (x=Grad(Gx_pre, dur_pe), y=Grad(Gy_k, dur_pe))               # 4. prewinder + PE
        te1_d > 1e-9 && (seq += Delay(te1_d))                               # 5. TE/2
        seq = apply_spoiler(seq, spoiler)                                   # 6a. pre-crusher: kills FID from 180°
        seq += (RF(amp_T, d_ref),)                                          # 6b. 180° refocus
        seq = apply_spoiler(seq, spoiler)                                   # 6c. post-crusher: rephases echo
        te2_d > 1e-9 && (seq += Delay(te2_d))                               # 7. TE/2 to echo
        seq += (ADC(Nfe, dur_adc), x=Grad(Gx_ro, dur_adc))                  # 8. readout
        seq = apply_spoiler(seq, spoiler)                                   # 9a. TR spoiler: kills residual Mxy
        tr_d > 1e-9 && (seq += Delay(tr_d))                                 # 9b. TR recovery
    end
    seq
end

"""
    se_2d_sequence(TE, TR; α_exc, FOV, Nfe, Npe, amp_T, spoiler) → Sequence

Multi-shot 2D Cartesian spin-echo sequence — `ir_se_2d_sequence` with the
inversion + TI block dropped. Used for spatially-resolved **T2** mapping: sweep
`TE` across acquisitions and fit `exp(−TE/T2)` per voxel. `Npe` shots, one
`Nfe`-sample readout each; same gradient/k-space convention, pixel mapping, and
reconstruction (`abs.(ifft(ksp, (1,2)))`) as `ir_se_2d_sequence`.

The 180° refocus removes T2* (so the echo decays as the clean T2, not T2*);
`TR ≫ T1` makes each shot start from a recovered M0 so the T1 weighting drops
out and only the `exp(−TE/T2)` dependence remains.
"""
function se_2d_sequence(TE::Real, TR::Real;
                        α_exc::Real             = π / 2,
                        FOV::Real               = 0.2,
                        Nfe::Int                = 16,
                        Npe::Int                = 8,
                        amp_T::Real             = 20e-6,
                        spoiler::SpoilerConfig  = SpoilerConfig())
    γ_Hz = 42.577e6   # proton gyromagnetic ratio [Hz/T], Koma convention

    d_exc = rf_duration(α_exc; amp_T = amp_T)
    d_ref = rf_duration(π;     amp_T = amp_T)

    dur_adc = 1e-3            # readout window [s]
    dur_pe  = dur_adc / 2     # prewinder / phase-encode duration [s]

    kmax_x = Nfe / (2.0 * FOV)
    Gx_pre = kmax_x / (γ_Hz * dur_pe)
    Gx_ro  = 2.0 * kmax_x / (γ_Hz * dur_adc)

    Δky      = 1.0 / FOV
    ky_steps = [(k - 1 - Npe ÷ 2) * Δky for k in 1:Npe]

    d_crush = spoiler.enabled ? spoiler.dur : 0.0

    te1_d = max(TE / 2 - d_exc / 2 - dur_pe - d_crush - d_ref / 2, 0.0)
    te2_d = max(TE / 2 - d_ref / 2 - d_crush - dur_adc / 2, 0.0)

    shot_time = d_exc + dur_pe + te1_d +
                d_crush + d_ref + d_crush + te2_d + dur_adc + d_crush
    tr_d      = max(TR - shot_time, 0.0)

    seq = Sequence()
    @addblocks for k in 1:Npe
        Gy_k = -ky_steps[k] / (γ_Hz * dur_pe)

        seq += (RF(amp_T, d_exc),)                                          # 1. excitation
        seq += (x=Grad(Gx_pre, dur_pe), y=Grad(Gy_k, dur_pe))               # 2. prewinder + PE
        te1_d > 1e-9 && (seq += Delay(te1_d))                               # 3. TE/2
        seq = apply_spoiler(seq, spoiler)                                   # 4a. pre-crusher
        seq += (RF(amp_T, d_ref),)                                          # 4b. 180° refocus
        seq = apply_spoiler(seq, spoiler)                                   # 4c. post-crusher
        te2_d > 1e-9 && (seq += Delay(te2_d))                               # 5. TE/2 to echo
        seq += (ADC(Nfe, dur_adc), x=Grad(Gx_ro, dur_adc))                  # 6. readout
        seq = apply_spoiler(seq, spoiler)                                   # 7a. TR spoiler
        tr_d > 1e-9 && (seq += Delay(tr_d))                                 # 7b. TR recovery
    end
    seq
end

"""
    ir_tse_2d_sequence(TI, esp, TR; etl, α_exc, FOV, Nfe, Npe, amp_T, spoiler) → Sequence

Inversion-recovery **turbo spin echo** (IR-TSE) — `ir_se_2d_sequence` with an
echo train of length `etl` after each excitation. A single inversion + excitation
is followed by `etl` 180° refocuses, each reading a *different* phase-encode line,
so one shot fills `etl` rows of k-space and the whole image needs only `Npe ÷ etl`
shots (≈`etl`× faster than the single-echo IR readout). `etl = 1` reproduces
`ir_se_2d_sequence(TI, esp, TR)` (echo at `TE = esp`); `etl` must divide `Npe`.

Echo `e` forms at `t = e·esp` after excitation. Phase-encode **view ordering is
linear**: shot `s` acquires lines `(s−1)·etl+1 … s·etl`, so the acquisition order
of ADC profiles equals the k-space line order and `raw_to_kspace` reconstructs
unchanged. The per-echo `Gy` is a blip *after* the 180° (set directly to
`+ky_steps[line]`) plus a rewind to `ky=0` *after* the readout — ADC always runs
with `Gy=0` (clean Cartesian), and `ky=0` at every refocus so the echo train is
undisturbed. `Gx` reuses the spin-echo convention: one positive prewinder; each
180° conjugates `kx`, each positive readout sweeps `−kmax→+kmax`.

T2 weighting varies across the echo train (later lines decay more), so the image
carries an *effective TE* set by the echo that fills the central k line — the
T2-blurring vs scan-time trade the agent can learn. Imperfect 180°s would add
stimulated echoes (EPG territory); the digital twin's exact 180°s keep the train
clean.
"""
function ir_tse_2d_sequence(TI::Real, esp::Real, TR::Real;
                            etl::Int                = 2,
                            α_exc::Real             = π / 2,
                            FOV::Real               = 0.2,
                            Nfe::Int                = 16,
                            Npe::Int                = 8,
                            amp_T::Real             = 20e-6,
                            spoiler::SpoilerConfig  = SpoilerConfig())
    etl >= 1 || error("etl must be ≥ 1; got $etl")
    Npe % etl == 0 || error("etl ($etl) must divide Npe ($Npe)")
    γ_Hz = 42.577e6

    d_inv = rf_duration(π;     amp_T = amp_T)
    d_exc = rf_duration(α_exc; amp_T = amp_T)
    d_ref = rf_duration(π;     amp_T = amp_T)

    dur_adc = 1e-3
    dur_pe  = dur_adc / 2

    kmax_x = Nfe / (2.0 * FOV)
    Gx_pre = kmax_x / (γ_Hz * dur_pe)
    Gx_ro  = 2.0 * kmax_x / (γ_Hz * dur_adc)

    Δky      = 1.0 / FOV
    ky_steps = [(k - 1 - Npe ÷ 2) * Δky for k in 1:Npe]

    d_crush = spoiler.enabled ? spoiler.dur : 0.0

    ti_d     = max(TI    - d_inv / 2 - d_exc / 2, 0.0)
    # exc → first 180° (prewinder dur_pe sits between excitation and 180°).
    te1_d    = max(esp/2 - d_exc/2 - dur_pe - d_crush - d_ref/2, 0.0)
    # 180° → echo (PE blip dur_pe sits between 180° and readout).
    te_mid_d = max(esp/2 - d_ref/2 - d_crush - dur_pe - dur_adc/2, 0.0)
    # echo → next 180° (PE rewind dur_pe sits between readout and next 180°).
    te_gap_d = max(esp/2 - dur_adc/2 - dur_pe - d_crush - d_ref/2, 0.0)
    (esp/2 - d_exc/2 - dur_pe - d_crush - d_ref/2) > 0 ||
        error("esp too short for excitation + prewinder + 180°; got esp=$esp")
    (esp/2 - d_ref/2 - d_crush - dur_pe - dur_adc/2) > 0 ||
        error("esp too short for 180° + PE blip + ADC; got esp=$esp")

    n_shots = Npe ÷ etl
    per_echo1 = te1_d    + d_crush + d_ref + d_crush + dur_pe + te_mid_d + dur_adc + dur_pe
    per_echoN = te_gap_d + d_crush + d_ref + d_crush + dur_pe + te_mid_d + dur_adc + dur_pe
    shot_time = d_inv + ti_d + d_exc + dur_pe + per_echo1 + (etl - 1) * per_echoN + d_crush
    tr_d      = max(TR - shot_time, 0.0)

    seq = Sequence()
    @addblocks for s in 1:n_shots
        seq += (RF(amp_T, d_inv),)                                          # 1. 180° inversion
        ti_d > 1e-9 && (seq += Delay(ti_d))                                 # 2. TI
        seq += (RF(amp_T, d_exc),)                                          # 3. excitation
        seq += (x = Grad(Gx_pre, dur_pe),)                                  # 4. kx prewinder (once)
        for e in 1:etl
            line    = (s - 1) * etl + e
            Gy_blip = ky_steps[line] / (γ_Hz * dur_pe)                      # +ky after the 180°
            pre_d   = (e == 1) ? te1_d : te_gap_d
            pre_d > 1e-9 && (seq += Delay(pre_d))                           # 5. TE/2 (exc/echo → 180°)
            seq = apply_spoiler(seq, spoiler)                               # 6a. pre-crusher
            # Meiboom–Gill: refocus about y (90° RF phase) while the excitation
            # is about x. Without it, non-MG 180°ₓ pulses alternate the echo sign
            # down the train → an alternating sign per ky line → half-FOV PE shift.
            seq += (RF(complex(0.0, amp_T), d_ref),)                        # 6b. 180°_y refocus (MG)
            seq = apply_spoiler(seq, spoiler)                               # 6c. post-crusher
            seq += (y = Grad(Gy_blip, dur_pe),)                            # 7. PE blip → +ky_steps[line]
            te_mid_d > 1e-9 && (seq += Delay(te_mid_d))                     # 8. 180° → echo
            seq += (ADC(Nfe, dur_adc), x = Grad(Gx_ro, dur_adc))           # 9. readout (Gy = 0)
            seq += (y = Grad(-Gy_blip, dur_pe),)                          # 10. PE rewind → ky = 0
        end
        seq = apply_spoiler(seq, spoiler)                                   # 11a. TR spoiler
        tr_d > 1e-9 && (seq += Delay(tr_d))                                 # 11b. TR recovery
    end
    seq
end

"""
    gre_2d_sequence(TE, TR; α, FOV, Nfe, Npe, amp_T, spoiler) → Sequence

Multi-shot 2D Cartesian gradient-echo sequence. `Npe` shots are concatenated,
each with a different Gy phase-encode amplitude. Returns a `Sequence` with
exactly `Npe` ADC blocks (one `Nfe`-sample readout per phase-encode step).

Differences from `ir_se_2d_sequence`:
  - No 180° refocus pulse: echo is formed by Gx polarity reversal alone.
  - Prewinder Gx is *negative* (walks kx → −kmax_x directly — no 180° to
    conjugate, unlike spin echo where the prewinder is positive and the
    refocus flips it).
  - Phase-encode Gy is *positive* (no sign flip needed).
  - Echo is T2*-weighted, not T2-weighted; decays faster than spin echo.

The TR spoiler is doing real work here: at short TR, residual Mxy from the
previous shot survives into the next α excitation and creates banding /
stimulated-echo contamination. Note this implements gradient spoiling only;
a fully spoiled GRE (FLASH) additionally varies the RF phase per shot
(quadratic increments) to scramble coherence pathways. Add later if needed.

After `simulate()`, reconstruct as `abs.(ifft(ksp, (1,2)))` with the same
pixel mapping as `ir_se_2d_sequence`.
"""
function gre_2d_sequence(TE::Real, TR::Real;
                          α::Real                = deg2rad(15),
                          FOV::Real              = 0.2,
                          Nfe::Int               = 16,
                          Npe::Int               = 8,
                          amp_T::Real            = 20e-6,
                          spoiler::SpoilerConfig = SpoilerConfig())
    γ_Hz = 42.577e6

    d_exc = rf_duration(α; amp_T = amp_T)

    dur_adc = 1e-3
    dur_pe  = dur_adc / 2

    kmax_x = Nfe / (2.0 * FOV)
    Gx_pre = -kmax_x / (γ_Hz * dur_pe)        # negative — see docstring
    Gx_ro  = 2.0 * kmax_x / (γ_Hz * dur_adc)

    Δky      = 1.0 / FOV
    ky_steps = [(k - 1 - Npe ÷ 2) * Δky for k in 1:Npe]

    d_crush = spoiler.enabled ? spoiler.dur : 0.0

    # TE measured from centre of α-pulse to centre of ADC (echo at kx=0).
    te_d = max(TE - d_exc / 2 - dur_pe - dur_adc / 2, 0.0)

    shot_time = d_exc + dur_pe + te_d + dur_adc + d_crush
    tr_d      = max(TR - shot_time, 0.0)

    seq = Sequence()
    @addblocks for k in 1:Npe
        # Positive Gy: no 180° to conjugate, so the echo samples ky directly.
        Gy_k = ky_steps[k] / (γ_Hz * dur_pe)

        seq += (RF(amp_T, d_exc),)                                          # 1. α excitation
        seq += (x=Grad(Gx_pre, dur_pe), y=Grad(Gy_k, dur_pe))               # 2. prewinder + PE
        te_d > 1e-9 && (seq += Delay(te_d))                                 # 3. TE
        seq += (ADC(Nfe, dur_adc), x=Grad(Gx_ro, dur_adc))                  # 4. readout (echo at midpoint)
        seq = apply_spoiler(seq, spoiler)                                   # 5. TR spoiler: kills residual Mxy
        tr_d > 1e-9 && (seq += Delay(tr_d))                                 # 6. TR recovery
    end
    seq
end

"""
    single_spin_phantom(; T1, T2, ρ = 1.0)

Zero-dimensional phantom (one spin at the origin) with the given
relaxation times. Fastest possible simulate-target — the right thing
for non-spatial E0 measurements, mirroring `01-FID.jl`.
"""
function single_spin_phantom(; T1::Real, T2::Real, ρ::Real = 1.0)
    Phantom(x = [0.0], T1 = [Float64(T1)], T2 = [Float64(T2)],
            ρ = [Float64(ρ)])
end
