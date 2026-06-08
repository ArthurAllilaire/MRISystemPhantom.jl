# Random Phantom Configuration Plan

## Context

The current library has two separate mechanisms that partially cover domain
randomisation:

- `PhantomConfig` is a deterministic build contract for one concrete phantom.
- `AugmentConfig` applies post-voxelisation, per-spin noise and dropout.

The E2 RL environment currently implements episode-level randomisation outside
the library in `_e2_build_episode_phantom`. It:

- takes a fixed pool of 14 T1-plate descriptors from `sphere_descriptors(:T1, base_cfg)`,
- samples a subset of those descriptors,
- jitters each sphere's T1 log-normally,
- derives T2 by preserving each sphere's nominal `T2/T1` ratio,
- rebuilds descriptors with `with_sphere_relaxation`,
- samples a restricted in-plane pose, and
- injects the result through `PhantomConfig.custom_sphere_descriptors`.

That works, but it leaves too much experiment-specific code in the RL project
and only handles one narrow policy: T1-plate spheres, log-normal T1 jitter, and
ratio-preserving T2.

### Why `AugmentConfig` is not the right tool for this

The natural question is whether `AugmentConfig` already covers "different random
T1/T2 per scenario". It does not, and understanding why motivates the whole
feature.

`apply_per_spin_noise!` (`src/augment.jl`) applies *independent per-spin*
Gaussian jitter:

```julia
obj.T1[i] = max(1e-6, obj.T1[i] * (1 + aug.T1_sigma_rel * randn(rng)))
```

Every voxel inside a sphere draws its own noise, so the sphere's *expected* T1
stays at the nominal value — the noise only blurs each sphere around its known
nominal. An agent can therefore still learn "the sphere at this position has
T1 ≈ nominal" and is never penalised; the true configuration has effectively
leaked.

The "by the book" requirement is the opposite: draw **one** random T1/T2 (and
optionally ρ) **per sphere per episode**, so the nominal values are never
observable and only the hidden `truth` record knows them. That is a
descriptor-level operation that must happen *before* voxelisation, which is
exactly what this API adds. `AugmentConfig` remains complementary: per-spin
measurement-style noise layered on top of an already-randomised episode.

## Goal

Add a high-level random phantom API that can sample complete deterministic
`PhantomConfig`s for training episodes while preserving the true sampled
configuration for evaluation and analysis.

The training code should be able to do roughly:

```julia
rpcfg = RandomPhantomConfig(...)
episode = sample_phantom_config(rpcfg; rng_seed = seed)

phantom = build_phantom(episode.cfg)
# Do not expose `episode.truth` to the agent during training.
```

The result should replace most of the custom phantom-building logic in
`_e2_build_episode_phantom`.

### Phasing

The feature splits into two phases so the supervisor's requirement is met
quickly and the heavy machinery is optional:

- **Phase 1 (MVP, required).** `RandomPhantomConfig`, `RandomPhantomEpisode`,
  the low-level `(rng, d, ctx) -> result` material hook, the sphere selector and
  pose sampler hooks, validation, the `truth` record, and the E2 migration using
  plain closures / small structs. This alone fully satisfies the "random T1/T2
  per episode, true config hidden for eval" goal.
- **Phase 2 (convenience, optional).** The declarative `MaterialDistributionSampler`
  DSL (`PerLabel` / `PerPlate` / `PerSphere` / `PreserveNominalRatio` /
  `ScaledFrom`). This is where most of the implementation and test cost lives
  (per-sphere dependency resolution, recursive container lookup). Build it only
  after Phase 1 has shown which declarative patterns are actually needed.

A closure passed to the low-level hook can express every Phase-2 policy
directly, so Phase 2 is strictly ergonomic.

## Non-Goals

- Do not turn `AugmentConfig` into a descriptor-level randomisation system.
  `AugmentConfig` should remain post-voxelisation, per-spin noise.
- Do not make `PhantomConfig` itself stochastic. A `PhantomConfig` should still
  represent one exact phantom.
- Do not hard-code one T1/T2 coupling policy into the builder. Ratio-preserving
  behaviour can be a convenience default, but users must be able to replace it.

## Proposed Types

### `RandomPhantomConfig`

`RandomPhantomConfig` describes a distribution over deterministic phantoms.

Proposed fields:

```julia
Base.@kwdef struct RandomPhantomConfig
    base::PhantomConfig = PhantomConfig()
    sphere_selector::Any = nothing
    material_sampler::Any = nothing
    pose_sampler::Any = nothing
    rng_seed::Int = 0
end
```

Notes:

- Fields are intentionally duck-typed at first. This keeps the interface open to
  `Distributions.jl` objects, custom functions, callable structs, and future
  helper sampler types.
- `base` holds deterministic build settings such as field strength, voxel size,
  water options, slice settings, and default included plates.
- The sampled result should be represented as an ordinary `PhantomConfig` with
  `custom_sphere_descriptors` filled in.
- There is no `include_truth` flag. `truth` is cheap to build and is always
  returned; *exposure* is gated by the RL environment, not by this config.
- `base` is the source of nominal descriptors, so it should have selection and
  per-spin randomisation **disabled** to avoid double randomisation: build
  nominal descriptors from a base with `augment.drop_sphere_p = 0`,
  `keep_sphere_labels = nothing`, and `drop_sphere_labels = []`. Selection is now
  the `sphere_selector`'s job, and per-spin noise is applied once, at build time,
  by the sampled config's own `augment`. See "Integration With Existing Builder".

### `RandomPhantomEpisode`

`sample_phantom_config` should return both the deterministic config and hidden
truth metadata:

```julia
struct RandomPhantomEpisode
    cfg::PhantomConfig
    truth::NamedTuple
end
```

The `truth` value should include enough information for evaluation:

```julia
(;
    descriptors_nominal,
    descriptors_sampled,
    active_labels,
    active_indices_by_plate,
    rotation,
    translation_mm,
    episode_seed,   # seed that drove selection/material/pose sampling
    build_seed,     # derived seed written into cfg.rng_seed for build-time noise
)
```

`descriptors_sampled` is the **descriptor-level, pre-augment** truth. If the
built phantom uses per-spin material augment (`AugmentConfig.T1_sigma_rel`,
`T2_sigma_rel`, `PD_sigma_abs` ≠ 0), individual spins no longer equal these
descriptor values. For "true phantom config" experiments, per-spin *material*
augment should usually be disabled so each sphere has one exact material; per-spin
*measurement* noise (position, B0) can stay on. See "RNG streams".

The RL environment can keep this metadata out of the observation during
training and use it for evaluation, debugging, or supervised diagnostics.

Why keep `truth` when the built `KomaMRI.Phantom` already contains `T1`, `T2`,
`ρ`, and positions?

The voxelised phantom stores per-spin arrays, not the episode-level descriptor
state. After voxelisation, water insertion, pose transforms, and optional
`AugmentConfig` noise, it is awkward and lossy to recover:

- which sphere labels were active,
- which nominal sphere each sampled sphere came from,
- descriptor-level sampled `T1`, `T2`, `T2s`, and `ρ`,
- plate membership,
- sampled pose,
- sampled subset indices,
- nominal values before sampling.

`truth` is the hidden episode record. Training code should not expose it to the
agent; evaluation code can use it as ground truth.

## Public API

### Sampling

```julia
episode = sample_phantom_config(rpcfg::RandomPhantomConfig; rng_seed = rpcfg.rng_seed)
episode = sample_phantom(rpcfg::RandomPhantomConfig; rng_seed = rpcfg.rng_seed)
```

`sample_phantom_config` returns `RandomPhantomEpisode`.

`sample_phantom` is a convenience wrapper:

```julia
episode = sample_phantom_config(rpcfg; rng_seed)
phantom = build_phantom(episode.cfg)
(; phantom, cfg = episode.cfg, truth = episode.truth)
```

### Material Sampling Contract

There should be two material-sampling layers:

1. A low-level function hook for maximum flexibility.
2. A high-level declarative sampler for common experiments.

The high-level declarative sampler is important because realistic experiments
often need different distributions per sphere. A single long function with many
`if ctx.label === ...` branches would be hard to read and easy to get wrong.

### Low-Level Material Sampler

The material sampler should accept:

```julia
(rng, d::SphereDescriptor, ctx) -> result
```

Arguments:

- `rng`: the episode-local random number generator. All random draws should use
  this object so that `rng_seed` gives reproducible episodes.
- `d`: the nominal `SphereDescriptor` for the current sphere before
  randomisation. It contains the sphere geometry, label, nominal `T1`, nominal
  `T2`, nominal `T2s`, nominal proton density `ρ`, and off-resonance.
- `ctx`: contextual metadata for this sampling call. It lets the same sampler
  branch on plate, index, label, field strength, or the base config without
  having to parse descriptor labels manually.

`ctx` is a `NamedTuple` such as:

```julia
(;
    plate,
    index,
    label = d.label,
    field = base.field,
    serial_number_class = base.serial_number_class,
    cfg = base,
)
```

The result can be:

- a `SphereDescriptor`,
- a `NamedTuple` containing any of `T1`, `T2`, `T2s`, `ρ`, `delta_w`,
- `nothing`, meaning keep the nominal descriptor unchanged.

Result-handling rules:

- **NamedTuple merge.** Only the named fields override the nominal descriptor;
  everything else (geometry, label, unspecified materials) is preserved.
- **`T2s` default.** If a result sets `T2` but not `T2s`, default `T2s = T2`,
  matching `with_sphere_relaxation(d, T1, T2; T2s = T2)` in `src/builder.jl`. If
  neither is set, keep the nominal `T2s`. Do **not** rely on anything downstream
  to clamp `T2s` to `T2`: `build_sphere` writes the descriptor `T2s` verbatim
  (`T2s = fill(d.T2s, n)`), and the `T2s = min(T2s, T2)` clamp in
  `apply_per_spin_noise!` only runs when `AugmentConfig.T2_sigma_rel > 0`. Because
  there is no unconditional clamp, validation enforces `T2s <= T2` by default (see
  Material Validation).
- **Full-descriptor return overrides everything.** Returning a `SphereDescriptor`
  replaces the *whole* descriptor, including `centre`, `radius`, and `label`. Use
  this only when you intend to move/relabel the sphere; the common, safe path is
  the NamedTuple, which never changes geometry.
- **`delta_w` overlaps `AugmentConfig.B0_sigma_Hz`.** Descriptor-level `delta_w`
  and per-spin B0 jitter both set off-resonance. Pick one as the canonical knob
  per experiment to avoid double-counting.

Examples:

```julia
using Distributions

material_sampler = (rng, d, ctx) -> begin
    if ctx.plate === :T1
        T1 = rand(rng, LogNormal(log(d.T1), 0.20))
        T2 = T1 * d.T2 / d.T1
        (; T1, T2, T2s = T2)
    elseif ctx.plate === :T2
        T2 = rand(rng, LogNormal(log(d.T2), 0.20))
        T1 = T2 * d.T1 / d.T2
        (; T1, T2, T2s = T2)
    else
        nothing
    end
end
```

Fully custom correlated sampling is also possible:

```julia
material_sampler = (rng, d, ctx) -> begin
    T1, T2 = rand(rng, my_joint_T1_T2_distribution)
    ρ = rand(rng, my_pd_distribution)
    (; T1, T2, T2s = T2, ρ)
end
```

This keeps `Distributions.jl` useful without requiring every policy to be
expressed as a distribution object. Any callable user function is valid.

### Declarative Material Sampler

Add a convenience sampler for the common case where users want to specify
distributions by property, plate, sphere index, or exact label.

Sketch:

```julia
material = MaterialDistributionSampler(
    T1 = PerLabel(
        :T1_4  => Uniform(0.10, 0.25),
        :T1_14 => Uniform(2.50, 3.00),
        default = Uniform(0.50, 1.50),
    ),
    T2 = PerPlate(
        :T1 => PreserveNominalRatio(:T2, :T1),
        :T2 => PerSphere([
            Uniform(0.010, 0.020),
            Uniform(0.020, 0.030),
            # ...
        ]),
        :PD => Normal(0.080, 0.020),
    ),
    ρ = Truncated(Normal(1.0, 0.05), 0.0, 1.0),
)
```

Then:

```julia
rpcfg = RandomPhantomConfig(
    base = PhantomConfig(include_plates = [:T1, :T2, :PD, :water]),
    material_sampler = material,
)
```

Each property field holds one *property spec*. A property spec is resolved
**recursively** for a given sphere. The terminal (leaf) spec types are:

- constants, e.g. `1.0`,
- `Distributions.jl` distributions, e.g. `Uniform(0.2, 3.0)`,
- property functions with the full context signature `(rng, d, ctx) -> value`,
- derived-value helpers, e.g. `PreserveNominalRatio(:T2, :T1)` or
  `ScaledFrom(:T2, multiplier)`.

Two distinct function contracts (do not conflate them):

- **Property function** — a leaf used directly as a property spec. It takes
  `(rng, d, ctx)` (the same context as the low-level material hook) and is
  evaluated by `_sample_property`, not by `_sample`.
- **Sample spec** — a *scalar* draw used where only a number is needed: a
  distribution leaf, a constant, or the `multiplier` of `ScaledFrom`. These go
  through `_sample(rng, x)` (see Distributions strategy), whose `Function`
  method is the single-argument `f(rng)`. A sample spec does **not** receive
  `d`/`ctx`.

So a bare `f(rng)` is a multiplier/scalar; a `(rng, d, ctx)` function is a full
property leaf. Pick the right one for the slot.

The container spec types select a sub-spec and then recurse into it:

- `PerLabel(:T1_4 => spec, ..., default = spec)` — branch on the sphere's exact
  label.
- `PerPlate(:T1 => spec, ..., default = spec)` — branch on the sphere's plate.
- `PerSphere([spec, spec, ...]; default = spec)` — dense form, branch on the
  sphere's **stable label index** (see below).
- `PerSphere(Dict(4 => spec, 14 => spec); default = spec)` — sparse form, for
  overriding only a few indices. (A `Vector` cannot have holes; use the `Dict`
  form, or `PerLabel`, for sparse overrides.)

Resolution is therefore "walk down whatever the user nested": a `PerPlate` may
contain a `PerSphere`, whose entry may be a `PreserveNominalRatio`, etc. At each
container level an optional `default` covers unmatched keys. If nothing matches
and no `default` is given, **keep the nominal descriptor value** for that
property. The old idea of a single fixed five-step precedence is replaced by this
recursive descent, because which of label / plate / index applies depends on how
the user nested the containers, not on a global priority order.

`PerSphere` indexing — important. `PerSphere([d1, d2, ...])` indexes by the
sphere's **stable label-derived index**, not by its position in the selected
subset. `:T1_14` always resolves to entry `14`, regardless of how many spheres
the selector picked or in what order. This keeps the distribution → sphere
mapping invariant under subset sampling. The dense (`Vector`) form may be
**shorter** than the plate's sphere count and the sparse (`Dict`) form may omit
indices; an index with no entry falls through to the container's `default`, then
to the nominal value. For just a few overrides prefer the `Dict` form or
`PerLabel`.

This allows sparse overrides without making every sphere explicit:

```julia
material = MaterialDistributionSampler(
    T1 = PerLabel(
        :T1_4  => Uniform(0.10, 0.25),
        :T1_14 => Uniform(2.50, 3.00),
        default = Uniform(0.50, 1.50),
    ),
    T2 = PreserveNominalRatio(:T2, :T1),
    ρ = Truncated(Normal(1.0, 0.05), 0.0, 1.0),
)
```

#### Derived helpers and property-evaluation order

Two helpers compute a property *from another property's already-sampled value*:

- `PreserveNominalRatio(:dst, :src)` for property `dst`:
  `dst = sampled[src] * (nominal[dst] / nominal[src])`.
  Example: `T2 = PreserveNominalRatio(:T2, :T1)` sets
  `T2 = sampled_T1 * (nominal_T2 / nominal_T1)`. Requires `sampled[src]`.
- `ScaledFrom(:src, multiplier_spec)`:
  `value = _sample(rng, multiplier_spec) * sampled[src]`.
  Example: `T1 = ScaledFrom(:T2, Truncated(Normal(10.0, 1.0), 0.1, Inf))` sets
  `T1 = draw * sampled_T2`. Requires `sampled[src]`.

Because of these, properties **cannot** be sampled independently — a property
may depend on another property of the *same* sphere, and the direction of the
dependency varies per plate. In the target example, a T1-plate sphere needs
`T1 → T2` (T2 preserves the ratio from T1), while a T2-plate sphere needs
`T2 → T1` (T1 is scaled from T2). No single fixed property order is correct for
all spheres.

The declarative sampler must therefore, **for each sphere independently**:

1. Resolve each of the four property specs (`ρ`, `T1`, `T2`, `T2s`) to a leaf.
2. Build a dependency graph whose edges are `src -> dst` whenever `dst`'s leaf is
   a derived helper referencing `src`.
3. Topologically sort the graph and evaluate properties in that order, so every
   derived helper reads an already-sampled source.
4. **Detect cycles** (e.g. `T1 = ScaledFrom(:T2, …)` together with
   `T2 = PreserveNominalRatio(:T2, :T1)`) and raise a clear error naming the
   sphere and the cyclic properties, rather than reading a stale/nominal value.
5. Merge the four results into one descriptor (applying the `T2s` default rule),
   then run material validation.

A naive "sample T1 then T2" implementation is incorrect: on the T2 plate it would
evaluate `ScaledFrom(:T2, …)` before `T2` exists and silently multiply by the
nominal T2. The per-sphere topological pass is the required behaviour, not an
optimisation.

Example matching the desired policy:

```julia
material = MaterialDistributionSampler(
    T1 = PerPlate(
        :T1 => PerSphere([
            Uniform(0.10, 0.20),
            Uniform(0.20, 0.30),
            # ...
            Uniform(2.50, 3.00),
        ]),
        :T2 => ScaledFrom(:T2, Truncated(Normal(10.0, 1.0), 0.1, Inf)),
        :PD => Truncated(Normal(1.0, 0.2), 1e-6, Inf),
    ),
    T2 = PerPlate(
        :T1 => PreserveNominalRatio(:T2, :T1),
        :T2 => PerSphere([
            Uniform(0.010, 0.020),
            Uniform(0.020, 0.030),
            # ...
        ]),
        :PD => Truncated(Normal(0.080, 0.020), 1e-6, Inf),
    ),
    ρ = Truncated(Normal(1.0, 0.05), 0.0, 1.0),
)
```

Here:

- T1-plate T1 values use per-sphere uniform distributions.
- T1-plate T2 values preserve each sphere's nominal `T2/T1` ratio.
- T2-plate T2 values use per-sphere uniform distributions.
- T2-plate T1 values are a positive Gaussian-like multiplier times sampled T2.
- PD-plate T1 and T2 are sampled independently of sampled PD.
- PD is sampled independently for all contrast spheres.

### Material Validation

Material sampling should fail fast when a sampler returns physically invalid or
numerically unusable values. Validation should run after applying the sampler and
before the descriptor is stored in the sampled deterministic `PhantomConfig`.

Default validation rules:

- `T1` must be finite and strictly positive.
- `T2` must be finite and strictly positive.
- `T2s` must be finite and strictly positive.
- `T2s <= T2` (with a small tolerance). Physical T2* is no longer than T2, and
  nothing downstream clamps it unconditionally (`build_sphere` writes `d.T2s`
  as-is), so an out-of-order `T2s` would otherwise reach the simulation silently.
  Make this toggleable for advanced synthetic experiments.
- `ρ` must be finite and non-negative.
- By default, contrast-sphere `ρ` should also be `<= 1.0`, because the current
  descriptor convention is `ρ = 1.0` for bulk water.
- `delta_w` must be finite.

Do not enforce `T1 > T2` or `T2 > T1`. `T1 > T2` is common and should be
accepted, but the API should also avoid rejecting unusual synthetic examples
unless they violate the basic positivity/finite checks above.

Potential optional / configurable checks:

- relax `T2s <= T2` for advanced synthetic experiments,
- configurable proton-density bounds, e.g. `ρ_bounds = (0.0, 1.0)` by default
  and `ρ_bounds = (0.0, Inf)` for advanced synthetic experiments.

Proposed user-facing error style:

```julia
Invalid sampled material for sphere :T2_4 on plate :T2:
T1 must be finite and > 0, got -0.12
```

This should make bad sampler definitions obvious, especially when distributions
can produce negative values, such as an unconstrained `Normal`.

### Sphere Selection Contract

The sphere selector controls which nominal descriptors become active in an
episode.

Recommended low-level contract:

```julia
(rng, descriptors_by_plate, base_cfg) -> selected_descriptors_by_plate
```

`descriptors_by_plate` is a dictionary or named collection containing the
nominal descriptors for each generated plate.

The selected result should preserve plate information so the material sampler
can receive useful context. Note that `SphereDescriptor` (`src/sphere_descriptor.jl`)
has **no `plate` field** — plate membership is carried by the
`descriptors_by_plate` grouping, and the stable per-sphere index used by
`PerSphere` is derived from the label suffix (`:T1_14` → plate `:T1`, index
`14`). The by-plate grouping is the source of truth for both `ctx.plate` and
`PerSphere` indexing.

Convenience helper types can be added on top. The selector count helper is named
`SphereCountPerPlate` to avoid confusion with the material-sampler container
`PerPlate` (different role: one picks *how many* spheres, the other picks *which
distribution*):

```julia
SphereCountPerPlate(
    :T1 => 6:14,
    :T2 => 0:14,
    :PD => 0:14,
)
```

Expected behaviour:

- `nothing`: keep all nominal descriptors from the **contrast** plates in
  `base.include_plates`.
- `Integer`: sample exactly that many spheres from all contrast plates.
- `UnitRange` or distribution-like object: sample a count, then sample that many
  spheres.
- `Dict{Symbol,Any}`: apply counts or selectors per plate.
- callable: use the low-level custom selector contract.

Plate-kind semantics (important — only `:T1`, `:T2`, `:PD`, `:fiducials` are
valid `sphere_descriptors` plates; `sphere_descriptors(:water, …)` errors with
"Unknown plate"):

- **`:water`** is not a sphere-descriptor plate. It is never passed to
  `sphere_descriptors` and never goes through selection/material sampling. It is
  carried straight through to the sampled config's `include_plates` so the
  background-water cutout still runs (see Integration).
- **`:fiducials`** stays **deterministic by default**, even though the default
  `PhantomConfig()` includes it. Fiducials are not selected or randomised unless
  the user explicitly opts them into the selector/material policy. By default they
  remain a generated plate in the sampled config's `include_plates`.
- Contrast plates (`:T1`/`:T2`/`:PD`) listed in `base.include_plates` are owned by
  the random pipeline: after sampling they are either present as sampled
  `custom_sphere_descriptors` or absent if the selector drops them. They are never
  silently regenerated as deterministic plates. Deterministic non-contrast
  generated plates (e.g. fiducials) stay in `include_plates`.

### Pose Sampling Contract

The pose sampler controls `PhantomConfig.rotation` and
`PhantomConfig.translation_mm`.

Low-level contract:

```julia
(rng, base_cfg) -> NamedTuple
```

The returned `NamedTuple` may contain:

```julia
(; rotation, translation_mm)
```

Convenience helpers can cover common policies:

```julia
InPlanePoseSampler(
    rotation_sigma_rad = 0.05,
    translation_sigma_mm = 2.0,
)

FullPoseSampler(
    rotation = something,
    translation_mm = something,
)
```

**Design tension to make explicit.** The supervisor's request is a "completely
randomly oriented" phantom, but the E2 environment needs an *in-plane-only*
option because out-of-plane tilt and z-translation move spheres out of the thin
axial slab it observes (`slice_thickness_mm`). These two goals conflict and the
choice is per-experiment, not global:

- **Sliced observation (current E2):** in-plane rotation + x/y translation only.
- **Volumetric observation:** full 3D orientation is meaningful and desirable.

Both must be selectable `pose_sampler`s, never hard-coded in the RL loop.

Start with Euler-angle pose samplers because they match `PhantomConfig.rotation`
and the existing `rotation_matrix` / `apply_transform!` pipeline.

Full uniform 3D orientation is a first-class goal but is **deferred** in the
first version. Sampling Euler angles uniformly does *not* produce a uniform
distribution over 3D orientations. True uniform orientation sampling draws a
random unit quaternion (uniform on `SO(3)`) and applies the corresponding
rotation matrix. It is cheap to add once `PhantomConfig` accepts either Euler
angles or a rotation matrix: add an `apply_transform!` method taking a 3×3 matrix
(the current method already builds one via `rotation_matrix`). When a volumetric
experiment needs it, add a `UniformSO3PoseSampler` rather than approximating with
Gaussian Euler angles. Until then, do not pretend Gaussian Euler perturbation is
uniform orientation.

Initial helper samplers should therefore focus on:

- fixed pose,
- Gaussian Euler-angle perturbations,
- in-plane rotation plus x/y translation,
- user-supplied callable pose samplers.

## T1/T2 Coupling Policy

Do not force one universal rule.

Support these patterns:

1. Preserve nominal ratio.

   Useful default when sampling one relaxation parameter and deriving the other:

   ```julia
   T2 = T1 * d.T2 / d.T1
   T1 = T2 * d.T1 / d.T2
   ```

2. Sample T1 and T2 independently.

   Useful for broad stress testing, but can generate unrealistic tissues unless
   distributions are chosen carefully.

3. Sample T1 and T2 jointly.

   Best for realistic correlated tissue distributions.

4. Recompute PD-plate relaxation from sampled proton density.

   For PD spheres, a helper may optionally sample `ρ` and reuse `pd_t1(ρ, field)`
   and `pd_t2(ρ, field)`.

   Caveat: as currently implemented (`src/materials/pd_array.jl`), `pd_t1` and
   `pd_t2` **ignore `ρ`** and return the constant background-water relaxation, so
   this coupling is a no-op today. It only becomes useful once those functions
   gain genuine ρ-dependence; until then, sample PD-plate T1/T2 directly. Do not
   ship this helper as a headline feature on top of the current stubs.

The core API should only provide the hooks. Convenience constructors can provide
ratio-preserving defaults.

## Distributions.jl Dependency Strategy

Add `Distributions.jl` as a hard dependency. This feature is explicitly about
sampling phantom parameters from distributions, and the report/user-facing
examples should be able to cite and use the standard Julia distributions library
directly.

Still keep the API duck-typed rather than forcing every sampler field to be a
`Distribution`. The canonical path should be `Distributions.jl`, but constants,
functions, and custom sampler objects should remain valid.

Internally use a small sampling helper for **scalar sample specs** (distribution
leaves, constants, `ScaledFrom` multipliers). It is *not* the path for full
`(rng, d, ctx)` property functions — those are dispatched by `_sample_property`:

```julia
_sample(rng, x) = rand(rng, x)
_sample(rng, x::Real) = x
_sample(rng, f::Function) = f(rng)   # scalar: single-arg f(rng) only
```

This supports:

- `Uniform(...)`,
- `Normal(...)`,
- `LogNormal(...)`,
- truncated distributions,
- constants,
- lambdas/functions,
- callable structs,
- custom objects that implement `Random.rand(rng, sampler)`.

Example:

```julia
rand(rng, sampler)
```

works for `Distributions.jl` objects and user-defined sampler objects, while the
specialised methods above make constants and functions convenient.

Do not type convenience-sampler fields as `Distribution` unless a specific
method genuinely needs that restriction. Prefer `Any` or a small documented
"sample specification" union so the public API stays flexible.

## Integration With Existing Builder

The existing deterministic build path should remain mostly unchanged:

```julia
build_phantom(cfg::PhantomConfig)
```

The random path should happen before voxelisation:

1. Build nominal descriptors for the candidate contrast plates (`:T1`/`:T2`/`:PD`
   entries in `base.include_plates`) using the existing
   `sphere_descriptors(plate, base_cfg)`. Do not call `sphere_descriptors(:water,
   …)` — it errors. Generate from a base that has selection/dropout **disabled**
   (`augment.drop_sphere_p = 0`, `keep_sphere_labels = nothing`,
   `drop_sphere_labels = []`) so the new selector is the only thing choosing
   spheres — otherwise the builder's existing dropout/keep/drop logic compounds
   with `sphere_selector` (see `sphere_descriptors` in `src/builder.jl`).
2. Apply sphere selection.
3. Apply material sampling to selected descriptors.
4. Apply pose sampling by setting `rotation` and `translation_mm` on the sampled
   deterministic config.
5. Build the sampled `PhantomConfig`:
   - move the selected/sampled contrast descriptors into
     `custom_sphere_descriptors`,
   - set `include_plates` to the non-contrast entries of `base.include_plates`
     (e.g. `:water` and, by default, `:fiducials`), so they are still generated
     deterministically and the water cutout still runs,
   - drop all candidate contrast plates from `include_plates`, including contrast
     plates where the selector returned zero spheres, so dropped plates are absent
     rather than regenerated deterministically.
6. Store sampled descriptors and metadata in `truth`.

#### RNG streams: episode sampling vs `build_phantom`

There are **two** independent RNG streams and they must be reconciled:

- The episode sampler uses an episode-local `rng` seeded from the episode's
  `rng_seed` (material, selection, and pose draws).
- `build_phantom(cfg)` re-seeds its *own* `MersenneTwister(cfg.rng_seed)`
  internally (`src/builder.jl`) for water voxelisation and the per-spin
  `AugmentConfig` noise.

If the sampled `episode.cfg.rng_seed` is left at the constant `base.rng_seed`,
every episode gets the **same** per-spin augment-noise realisation even though
the spheres differ. The sampler must therefore write a per-episode `build_seed`
into `episode.cfg.rng_seed`.

`build_seed` must be a **pure, stable function of the episode seed**, e.g. an
explicit integer-mixing function such as SplitMix64 with the sign bit cleared.
Do **not** derive it by drawing from the episode RNG *after* material/pose
sampling — otherwise adding or reordering a sampler draw would silently change the
build noise for the same episode seed. Record both seeds in `truth`
(`episode_seed`, `build_seed`).

Per-spin material augment vs descriptor truth: because `truth.descriptors_sampled`
is pre-augment, leaving `AugmentConfig` material sigmas non-zero means spins
deviate from the recorded truth. For "true phantom config" experiments, set
`T1_sigma_rel = T2_sigma_rel = PD_sigma_abs = 0` on the sampled config's
`augment`; keep only measurement-style noise (`position_sigma_mm`, `B0_sigma_Hz`)
if desired.

Important water interaction:

- The water cutout uses `all_sphere_descriptors(cfg)`.
- Therefore sampled contrast descriptors must be present in
  `custom_sphere_descriptors` before building water.
- The deterministic sampled config should avoid also generating the original
  nominal contrast plates, otherwise spheres would be duplicated.

For example, if `base.include_plates == [:T1, :water]`, the sampled config
should likely become:

```julia
PhantomConfig(
    ...,
    include_plates = [:water],
    custom_sphere_descriptors = sampled_T1_descriptors,
)
```

## RL Observation Considerations

Varying the number of spheres changes the observation shape if the environment
uses one observation channel per sphere.

The random phantom API can support variable sphere count, but the RL environment
must choose a compatible observation design:

- image-only observations,
- fixed maximum sphere pool with padding,
- active-mask observations,
- set/attention-based policy networks,
- or fixed count per experiment.

For the current E2 style, the practical migration path is probably:

1. Keep a fixed maximum descriptor pool.
2. Sample active descriptors.
3. Return `truth.active_labels` and `truth.active_indices_by_plate`.
4. Let the environment pad missing sphere estimates and include an active mask.

Padding/mask helpers should remain in the RL project for now. This phantom
library should provide the sampled descriptor metadata needed to build such
observations, but it should not own policy-network observation layout.

Example of what the RL project may build:

```julia
t1_estimates = [1.1, 0.9, 0.0, 0.0, ...]      # fixed max length
active_mask  = [true, true, false, false, ...]
```

The mask tells the policy or loss which entries correspond to active sampled
spheres.

## Train/Eval Seed Protocol

This is the heart of the "by the book" requirement: the agent must never be
trained on a configuration that is later used to evaluate it. The library makes
each episode a pure function of its seed, but the *protocol* that keeps train and
eval disjoint lives in the experiment code and must be stated explicitly.

Recommended protocol:

- **Disjoint seed ranges.** Reserve a fixed held-out set of seeds for evaluation
  (e.g. `eval_seeds = 1:200`) and draw all training seeds from a disjoint range
  (e.g. `train_seed >= 10_000`). Never sample a training seed from the eval set.
- **Fixed eval pool.** Materialise the evaluation episodes once from the held-out
  seeds and reuse the same pool across checkpoints, so improvements reflect
  generalisation rather than a shifting target. A helper such as
  `eval_episodes(rpcfg, seeds) -> Vector{RandomPhantomEpisode}` makes this
  explicit.
- **Truth is eval-only.** During training the agent sees only the built phantom
  (image/observation); `episode.truth` is used solely by the evaluation/metrics
  code as ground truth, never in the observation or reward shaping that the
  policy can exploit.

The library guarantees reproducibility per seed; this section is the usage
contract that turns that into a clean train/eval split.

## Implementation Steps

### Phase 1 (MVP)

1. Add a new source file, likely `src/random_phantom.jl`.
2. Define `RandomPhantomConfig` (no `include_truth` field) and
   `RandomPhantomEpisode`.
3. Add internal helpers:
   - `_nominal_descriptors_by_plate(base_cfg)` — builds nominal descriptors with
     selection/dropout disabled and keeps plate grouping,
   - `_apply_sphere_selector`,
   - `_apply_material_sampler` (low-level hook),
   - `_apply_pose_sampler`,
   - `_descriptor_with_material` — applies NamedTuple/descriptor results with the
     `T2s` default rule,
   - `_episode_build_seed(rng_seed)` — deterministically derive the
     `episode.cfg.rng_seed` used for build-time water/augment noise,
   - `_sample(rng, x)` duck-typing helper.
4. Implement material validation with useful label/plate-specific errors.
5. Implement `sample_phantom_config` (builds nominal → select → sample materials
   → pose → assemble deterministic `PhantomConfig` with custom descriptors and a
   derived build seed → `truth`).
6. Implement `sample_phantom`.
7. Export the Phase-1 public types and functions from `src/MRISystemPhantom.jl`.
8. Add `Distributions.jl` to `Project.toml` and compatibility bounds.
9. Migrate `_e2_build_episode_phantom` to use the low-level hook (closures / small
   structs such as `E2SphereSelector`, `RatioPreservingLogNormalT1`,
   `InPlanePoseSampler`, `FixedPose`).
10. Add docs to `docs/src/phantom.md` or a new randomisation docs page.

### Phase 2 (declarative DSL, optional)

11. Define the declarative material helper types:
    - `MaterialDistributionSampler`,
    - `PerLabel`, `PerPlate`, `PerSphere` (container specs),
    - `PreserveNominalRatio`, `ScaledFrom` (derived-value leaf specs).
12. Add resolution helpers:
    - `_resolve_property_sampler` — recursive descent through containers to a leaf
      for a given sphere (stable label-index for `PerSphere`),
    - `_sample_property` — evaluate a leaf spec,
    - `_property_dependencies` + `_topo_order` — per-sphere dependency graph and
      topological order over `(ρ, T1, T2, T2s)`, with cycle detection that errors
      naming the sphere and cyclic properties.
13. Wire `MaterialDistributionSampler` into the same `_apply_material_sampler`
    path so it reuses validation and the `T2s` default.

## Test Plan

Phase 1 core tests:

- Same `RandomPhantomConfig` and `rng_seed` produce identical sampled configs
  *and* identical built phantoms (build-seed determinism).
- Different seeds produce different sampled material values or selections.
- Different seeds produce different per-spin augment-noise realisations (the
  derived `episode.cfg.rng_seed` varies and equals `truth.build_seed`).
- `build_seed` is a pure function of `episode_seed`: it is unchanged when the
  material/pose samplers are swapped for ones that draw a different number of
  random values (no dependence on RNG state after sampling).
- `:water` is never passed to `sphere_descriptors`; it stays in the sampled
  config's `include_plates` and the water cutout uses the sampled descriptors.
- Fiducials remain deterministic by default: with a default-ish `base`, fiducial
  descriptors are identical across seeds and are not in `custom_sphere_descriptors`.
- Validation rejects `T2s > T2` by default and the message names the sphere/plate.
- `sphere_selector = nothing` preserves all included nominal spheres.
- Per-plate sphere counts produce the requested active count.
- No double randomisation: a `base` with `drop_sphere_p > 0` does not cause extra
  dropout on top of the selector (nominal generation disables it).
- Material sampler returning `nothing` preserves descriptors.
- Material sampler returning `NamedTuple` updates only specified fields; setting
  `T2` without `T2s` defaults `T2s = T2`; setting neither keeps nominal `T2s`.
- Material sampler returning a full `SphereDescriptor` overrides geometry/label.
- Invalid sampled materials, such as negative T1, negative T2, negative T2s, or
  out-of-range ρ, throw clear errors containing the sphere label and plate.
- Pose sampler updates `rotation` and `translation_mm`.
- Sampled configs do not duplicate generated plates and custom descriptors.
- Water cutouts use the sampled custom descriptors.
- `truth.active_labels` / `active_indices_by_plate` match the selected spheres.

Train/eval protocol tests:

- Episodes built from the held-out eval seeds are reproducible and disjoint from
  any training seed (a fixed eval pool reproduces byte-for-byte across calls).

Phase 2 declarative-DSL tests:

- `MaterialDistributionSampler` supports a global distribution for each property.
- `PerLabel` overrides exact labels such as `:T1_14`.
- `PerPlate` applies different policies to `:T1`, `:T2`, and `:PD`.
- `PerSphere` indexes by the **stable label index** (`:T1_14` → entry 14), and
  the mapping is invariant under subset selection (selecting `{:T1_3, :T1_9}`
  still applies entries 3 and 9, not 1 and 2).
- Sparse/short `PerSphere` arrays fall through to the container `default`, then
  to the nominal value.
- Recursive resolution: a `PerPlate` whose branch is a `PerSphere` resolves
  correctly.
- `PreserveNominalRatio(:T2, :T1)` derives T2 from sampled T1 using the nominal
  `T2/T1` ratio; `ScaledFrom(:T2, m)` sets `value = sample(m) * sampled_T2`.
- Per-sphere evaluation order: a config that needs `T1 → T2` on the T1 plate and
  `T2 → T1` on the T2 plate produces correct derived values on both (no stale
  nominal reads).
- Cycle detection: `T1 = ScaledFrom(:T2, …)` with
  `T2 = PreserveNominalRatio(:T2, :T1)` raises a clear error naming the sphere and
  the cyclic properties.

RL-oriented tests:

- Reproduce the current E2 policy:
  - fixed T1 pool,
  - subset sampling,
  - log-normal T1 jitter,
  - T2 derived by nominal ratio,
  - in-plane pose.
- Verify that the sampled descriptor labels and active indices match the truth
  metadata.

Distributions-oriented tests:

- Use `Distributions.LogNormal` or `Distributions.Uniform` in a material sampler.
- Confirm the package can call `rand(rng, dist)` through duck typing.

## Migration Sketch For E2

Current E2 code can be reduced from manual descriptor reconstruction to:

```julia
rpcfg = RandomPhantomConfig(
    base = PhantomConfig(
        field = env.cfg_field,
        voxel_size_mm = env.voxel_size_mm,
        water_voxel_size_mm = env.water_voxel_size_mm,
        include_plates = env.include_water ? [:T1, :water] : [:T1],
        augment = AugmentConfig(B0_sigma_Hz = 5.0),
        slice_thickness_mm = env.voxel_size_mm,
        slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1),
    ),
    sphere_selector = E2SphereSelector(env.subset_size, forced_indices),
    material_sampler = RatioPreservingLogNormalT1(env.T1_sigma_rel),
    pose_sampler = env_cache_globally ? FixedPose() :
        InPlanePoseSampler(env.rotation_sigma_rad, env.translation_sigma_mm),
)

episode = sample_phantom_config(rpcfg; rng_seed)
phantom_cfg = episode.cfg
active_descs = episode.truth.descriptors_sampled
```

This keeps RL-specific policy in small sampler objects while moving the common
sampling and deterministic config construction into the library.

## Settled Decisions

These decisions are settled for the first implementation:

- Add `Distributions.jl` as a hard dependency.
- Use `Distributions.jl` in built-in convenience samplers, docs, and report
  examples.
- Keep sampler fields duck-typed so constants, functions, callable structs, and
  custom `rand(rng, sampler)` objects are accepted.
- Keep `truth` as a `NamedTuple` initially. Convert it to a concrete struct only
  if downstream code starts depending on a stable metadata schema.
- Keep variable-sphere padding/mask helpers in the RL project. The phantom
  library only needs to expose active labels, active indices, and sampled
  descriptors.
- Start with Euler-angle pose samplers because they match `PhantomConfig`.
  Defer true uniform `SO(3)` sampling, but treat it as a first-class
  `UniformSO3PoseSampler` to add later (via an `apply_transform!` matrix method),
  not an approximation with Gaussian Euler angles.
- Keep fiducials deterministic by default. Randomise fiducials only when the
  user explicitly includes/selects them in the randomisation policy.
- Ship in two phases: the low-level hooks + E2 migration first (Phase 1), the
  declarative `MaterialDistributionSampler` DSL second (Phase 2). A closure on the
  low-level hook can express every Phase-2 policy, so Phase 2 is purely
  ergonomic.
- Drop the `include_truth` config flag; always return `truth` and gate exposure
  in the RL layer.
- The sampler derives `episode.cfg.rng_seed` (the `build_seed`) as a pure,
  stable function of the episode seed so build-time water/augment noise varies
  per episode and stays reproducible; both `episode_seed` and `build_seed` are
  recorded in `truth`.
- Nominal descriptors are generated from a base with selection/dropout disabled
  so the new `sphere_selector` is the only selection mechanism.
- Train/eval disjointness is a usage protocol (held-out eval seed pool), not
  enforced by the library; document it and provide an `eval_episodes` helper.
- The declarative DSL resolves property specs by recursive descent and evaluates
  each sphere's properties in per-sphere topological order with cycle detection;
  `PerSphere` indexes by stable label index.
