"""
    AugmentConfig

Per-spin noise and sphere dropout applied after voxelisation. All fields
default to zero (no augmentation). Pass a non-zero `AugmentConfig` as the
`augment` field of [`PhantomConfig`](@ref) to enable domain randomisation
for RL training.

| Field | Default | Description |
|-------|---------|-------------|
| `T1_sigma_rel` | 0.0 | fractional Gaussian jitter on T1 (e.g. 0.05 → ±5%) |
| `T2_sigma_rel` | 0.0 | fractional Gaussian jitter on T2 |
| `PD_sigma_abs` | 0.0 | absolute Gaussian jitter on proton density ρ |
| `position_sigma_mm` | 0.0 | per-spin position noise standard deviation (mm) |
| `B0_sigma_Hz` | 0.0 | per-spin off-resonance standard deviation (Hz) |
| `drop_sphere_p` | 0.0 | Bernoulli probability of dropping each sphere |
"""
Base.@kwdef struct AugmentConfig
    T1_sigma_rel::Float64       = 0.0   # fractional Gaussian jitter on T1
    T2_sigma_rel::Float64       = 0.0   # fractional Gaussian jitter on T2
    PD_sigma_abs::Float64       = 0.0   # absolute Gaussian jitter on ρ
    position_sigma_mm::Float64  = 0.0   # per-spin position noise (mm)
    B0_sigma_Hz::Float64        = 0.0   # per-spin off-resonance stdev (Hz)
    drop_sphere_p::Float64      = 0.0   # Bernoulli prob of dropping a sphere
end

"""
    PhantomConfig

Single contract between the RL training loop and the builder. Everything the
builder needs to construct a phantom lives in this struct — there is no
hidden state.
"""
Base.@kwdef struct PhantomConfig
    field::Symbol                     = :T15           # :T15 or :T3
    voxel_size_mm::Float64            = 2.0            # isotropic voxel edge
    include_plates::Vector{Symbol}    = [:T1, :T2, :PD, :fiducials, :water]
    serial_number_class::Symbol       = :new           # :new (≥0042) or :legacy
    temperature_C::Float64            = 20.0
    rotation::NTuple{3,Float64}       = (0.0, 0.0, 0.0)   # Euler XYZ rad
    translation_mm::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    augment::AugmentConfig            = AugmentConfig()
    rng_seed::Int                     = 0
    custom_sphere_map::Dict{Symbol,Any} = Dict{Symbol,Any}()
    keep_sphere_labels::Union{Nothing,Vector{Symbol}} = nothing
    drop_sphere_labels::Vector{Symbol} = Symbol[]
    custom_sphere_descriptors::Vector{SphereDescriptor} = SphereDescriptor[]
    # Optional water-only voxel size. When provided, background water uses this
    # spacing and spin ρ is reweighted to conserve total water signal.
    water_voxel_size_mm::Union{Nothing,Float64} = nothing
    # Optional through-plane spacing for sliced water plane stacks. When
    # nothing, a sliced water slab is represented by one weighted centre plane.
    water_throughplane_voxel_size_mm::Union{Nothing,Float64} = nothing
    # Phantom-attached slab mask: when slice_thickness_mm !== nothing, keep only
    # spins whose signed distance from the phantom-frame plane is within half
    # the slab thickness. The plane passes through slice_center_mm and has
    # normal slice_normal. With the default normal, setting slice_center_mm to
    # (0, 0, PLATE_Z_MM.T1) isolates the T1 plate as before.
    # If slice_thickness_mm < voxel_size_mm the slab may end up empty.
    slice_thickness_mm::Union{Nothing,Float64} = nothing
    slice_center_mm::NTuple{3,Float64}         = (0.0, 0.0, 0.0)
    slice_normal::NTuple{3,Float64}            = (0.0, 0.0, 1.0)
end

"""
    scanner_for_field(field::Symbol) -> Scanner
    scanner_for_field(cfg::PhantomConfig) -> Scanner

Return a `Scanner` whose `B0` matches the phantom field strength: 3.0 T for
`:T3`, 1.5 T for `:T15`. Use this instead of bare `Scanner()` whenever the
phantom was built with a `PhantomConfig` so that the Larmor frequency in the
Bloch simulation is consistent with the relaxation values.
"""
scanner_for_field(field::Symbol) = Scanner(B0 = field === :T3 ? 3.0 : 1.5)
scanner_for_field(cfg::PhantomConfig) = scanner_for_field(cfg.field)
