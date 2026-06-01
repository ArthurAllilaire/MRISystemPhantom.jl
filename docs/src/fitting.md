# Parameter Fitting

The fitters recover quantitative tissue parameters from simulated (or
experimental) signal vectors. They are pure Julia — no external optimiser
dependency.

## T2 fitting

### `fit_t2_se` — log-linear SE fit

```julia
TEs  = [0.01, 0.02, 0.04, 0.08, 0.16]   # seconds
mags = [0.9,  0.75, 0.53, 0.28, 0.08]
result = fit_t2_se(TEs, mags)
result.T2        # s
result.S0        # amplitude
result.residual  # RMS log-space residual
```

Fits `|S| = S0 · exp(−TE/T2)` by log-linear regression. Requires ≥ 2
positive-magnitude samples.

## T1 fitting

### `fit_t1_ir` — 3-parameter magnitude IR

```julia
result = fit_t1_ir(TIs, mags)
result.T1        # s
result.A         # amplitude
result.B         # inversion efficiency
result.residual
```

Fits `|A − B · exp(−TI/T1)|`. Uses a log-spaced T1 grid with closed-form
(A, B) at each candidate — no iterative solver needed. Requires ≥ 3 samples.

### `fit_t1_generalized_ir` — generalised IR with uncertainty

The main fitter, supporting variable inversion angle, excitation angle,
TR, and finite-Npe transient models:

```julia
result = fit_t1_generalized_ir(
    TIs, αs, mags;
    TRs            = nothing,     # per-sample TR; nothing → ∞
    α_excs         = nothing,     # per-sample excitation angle; nothing → π/2
    Npe            = nothing,     # nothing → steady-state; Int → transient ramp
    T1_range       = (5e-3, 5.0), # search range (s)
    n_grid         = 300,
    abs_noise_sigma = 0.01,        # absolute noise σ (preferred)
    sigma_method   = :profile_likelihood,  # :asymptotic | :profile_likelihood | :bootstrap
)
result.T1
result.T1_sigma   # uncertainty estimate
```

**Forward model selection:**

| `Npe` | Model used |
|-------|------------|
| `nothing` | `steady_state_mz_at_excite` — infinite TR limit |
| `Int` | `transient_mz_at_excite_npe` — transient ramp over `Npe` shots |

Use `Npe = Npe` when fitting data from `ir_se_2d_sequence` (which starts
each acquisition from thermal equilibrium, not steady state).

**Uncertainty methods:**

| `sigma_method` | Notes |
|---------------|-------|
| `:asymptotic` | Cramér–Rao from Jacobian. Fast; overconfident on multimodal SSE. |
| `:profile_likelihood` | Span of T1 candidates within `σ²_eff` of minimum. Captures multimodal basins. Recommended. |
| `:bootstrap` | Residual bootstrap. Correct for any SSE landscape; slowest. |

**Noise floor:**

`abs_noise_sigma` (absolute, in signal units) is preferred over the legacy
`noise_sigma` (relative to data RMS). Providing it decouples the uncertainty
estimate from the agent's choice of actions.

### `fit_t1_t2_generalized_ir` — joint T1 + T2

Fits both T1 and T2 jointly from an IR-TSE acquisition where TE varies
across the echo train:

```julia
result = fit_t1_t2_generalized_ir(
    TIs, αs, TEs, mags;
    T1_range = (5e-3, 5.0),
    T2_range = (1e-3, 3.0),
    n_grid_t1 = 200,
    n_grid_t2 = 120,
    abs_noise_sigma = 0.01,
)
result.T1; result.T2
result.T1_sigma; result.T2_sigma   # profile-likelihood uncertainties
```

A 2-D log-grid scan; closed-form A at each (T1, T2) point. Profile-likelihood
σ on each axis (marginalised over the other).

## Analytical steady-state models

These are the forward models the fitters invert, exposed for use in
loss functions or simulating expected signals:

```julia
# Mz at excitation (steady-state)
Mz = steady_state_mz_at_excite(T1, TI, TR, θ_inv, α_exc)

# Mz at excitation (transient ramp from M0=1, averaged over Npe shots)
Mz = transient_mz_at_excite_npe(T1, TI, TR, θ_inv, α_exc; Npe = 64)

# Full per-shot Mz vector
mz_vec = transient_mz_per_shot(T1, TI, TR, θ_inv, α_exc, Npe)
```
