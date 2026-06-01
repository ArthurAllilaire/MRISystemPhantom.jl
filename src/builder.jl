"Fiducial sphere radius in metres (10 mm OD)."
const FIDUCIAL_RADIUS_M = 5.0e-3
"Contrast sphere radius in metres (15 mm OD)."
const CONTRAST_RADIUS_M = 7.5e-3
"Water housing sphere radius in metres (200 mm ID hemispheres joined)."
const HOUSING_RADIUS_M = 100e-3
const MM_TO_M = 1e-3
const PLATE_RNG_BUCKETS = 10_000

_mm_to_m(x) = x * MM_TO_M

"""
    with_sphere_relaxation(d, T1, T2; T2s=T2)

Return a copy of sphere descriptor `d` with updated relaxation values.
Geometry, density, off-resonance, and label are preserved.
"""
function with_sphere_relaxation(d::SphereDescriptor, T1::Real, T2::Real;
                                T2s::Real = T2)
    SphereDescriptor(
        d.centre, d.radius, d.ρ,
        Float64(T1), Float64(T2), Float64(T2s), d.delta_w, d.label,
    )
end

"""
    transform_descriptor(d, euler, translation)

Return a copy of sphere descriptor `d` whose centre has been rotated by
Euler angles `euler` and translated by `translation` (metres). Material
properties and radius are unchanged.
"""
function transform_descriptor(d::SphereDescriptor,
                              euler::NTuple{3,<:Real},
                              translation::NTuple{3,<:Real})
    transform_descriptor(d, rotation_matrix(euler...), translation)
end

function transform_descriptor(d::SphereDescriptor,
                              R::AbstractMatrix{<:Real},
                              translation::NTuple{3,<:Real})
    c_vec = R * collect(d.centre)
    c = (
        Float64(c_vec[1] + translation[1]),
        Float64(c_vec[2] + translation[2]),
        Float64(c_vec[3] + translation[3]),
    )
    SphereDescriptor(c, d.radius, d.ρ, d.T1, d.T2, d.T2s, d.delta_w, d.label)
end

function transform_descriptors(descs::AbstractVector{<:SphereDescriptor},
                               euler::NTuple{3,<:Real},
                               translation::NTuple{3,<:Real})
    [transform_descriptor(d, euler, translation) for d in descs]
end

function transform_descriptors(descs::AbstractVector{<:SphereDescriptor},
                               R::AbstractMatrix{<:Real},
                               translation::NTuple{3,<:Real})
    [transform_descriptor(d, R, translation) for d in descs]
end

function _t1_table(field::Symbol, class::Symbol)
    class === :legacy ? T1_ARRAY_LEGACY[field] : T1_ARRAY[field]
end

function _build_t1_descriptors(cfg::PhantomConfig)
    t1 = _t1_table(cfg.field, cfg.serial_number_class)
    t2 = T2_OF_T1_ARRAY[cfg.field]
    centres = contrast_plate_centres(PLATE_Z_MM.T1)
    descs = SphereDescriptor[]
    for i in eachindex(centres)
        push!(descs, SphereDescriptor(
            centres[i], CONTRAST_RADIUS_M, 1.0,
            t1[i], t2[i], t2[i], 0.0, Symbol("T1_$i")))
    end
    descs
end

function _build_t2_descriptors(cfg::PhantomConfig)
    t2 = T2_ARRAY[cfg.field]
    centres = contrast_plate_centres(PLATE_Z_MM.T2)
    descs = SphereDescriptor[]
    for i in eachindex(centres)
        push!(descs, SphereDescriptor(
            centres[i], CONTRAST_RADIUS_M, 1.0,
            T1_OF_T2_ARRAY[cfg.field][i], t2[i], t2[i], 0.0, Symbol("T2_$i")))
    end
    descs
end

function _build_pd_descriptors(cfg::PhantomConfig)
    centres = contrast_plate_centres(PLATE_Z_MM.PD)
    descs = SphereDescriptor[]
    for i in eachindex(centres)
        ρ = PD_FRACTIONS[i]
        T1 = pd_t1(ρ, cfg.field)
        T2 = pd_t2(ρ, cfg.field)
        push!(descs, SphereDescriptor(
            centres[i], CONTRAST_RADIUS_M, ρ,
            T1, T2, T2, 0.0, Symbol("PD_$i")))
    end
    descs
end

function _build_fiducial_descriptors(cfg::PhantomConfig)
    props = FIDUCIAL_PROPS[cfg.field]
    centres = fiducial_grid_centres()
    descs = SphereDescriptor[]
    for (i, c) in enumerate(centres)
        push!(descs, SphereDescriptor(
            c, FIDUCIAL_RADIUS_M, props.ρ,
            props.T1, props.T2, props.T2, 0.0, Symbol("fid_$i")))
    end
    descs
end

"""
    sphere_descriptors(plate, cfg)

Return the descriptor list for one plate, applying `drop_sphere_p` and any
entries from `cfg.custom_sphere_map` keyed on the sphere's label.
"""
function sphere_descriptors(plate::Symbol, cfg::PhantomConfig;
                            rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))::Vector{SphereDescriptor}
    descs = plate === :T1        ? _build_t1_descriptors(cfg) :
            plate === :T2        ? _build_t2_descriptors(cfg) :
            plate === :PD        ? _build_pd_descriptors(cfg) :
            plate === :fiducials ? _build_fiducial_descriptors(cfg) :
            error("Unknown plate $plate")

    # Custom override per label
    if !isempty(cfg.custom_sphere_map)
        descs = map(descs) do d
            get(cfg.custom_sphere_map, d.label, d)
        end
    end

    if cfg.keep_sphere_labels !== nothing
        keep_labels = Set(cfg.keep_sphere_labels)
        descs = [d for d in descs if d.label in keep_labels]
    end

    if !isempty(cfg.drop_sphere_labels)
        drop_labels = Set(cfg.drop_sphere_labels)
        descs = [d for d in descs if !(d.label in drop_labels)]
    end

    # Sphere dropout (only if p > 0)
    p = cfg.augment.drop_sphere_p
    if p > 0
        keep = [rand(rng) > p for _ in descs]
        descs = descs[keep]
    end

    descs
end

function _generated_sphere_descriptors(cfg::PhantomConfig;
                                       rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))
    descs = SphereDescriptor[]
    for plate in (:T1, :T2, :PD, :fiducials)
        plate ∈ cfg.include_plates || continue
        append!(descs, sphere_descriptors(plate, cfg; rng))
    end
    descs
end

"""
    all_sphere_descriptors(cfg)

Every contrast + fiducial sphere that the full phantom would contain.
Used by `build_background_water` to cut sphere volumes out of water.
"""
function all_sphere_descriptors(cfg::PhantomConfig;
                                rng::AbstractRNG = Random.MersenneTwister(cfg.rng_seed))
    descs = _generated_sphere_descriptors(cfg; rng)
    append!(descs, cfg.custom_sphere_descriptors)
    descs
end

"""
    sphere_descriptor_pixel(d, Npe, Nfe, FOV) -> (i_pe, i_fe)

Map a descriptor centre to the corresponding phase/frequency image pixel.
"""
function sphere_descriptor_pixel(d::SphereDescriptor, Npe::Int, Nfe::Int,
                                 FOV::Real)::NTuple{2,Int}
    (phys_to_pixel(d.centre[2], Npe, FOV),
     phys_to_pixel(d.centre[1], Nfe, FOV))
end

"""
    sphere_descriptor_pixels(descs, Npe, Nfe, FOV) -> Vector{NTuple{2,Int}}

Map a vector of sphere descriptors to image pixel indices `(i_pe, i_fe)`.
Convenience wrapper over [`sphere_descriptor_pixel`](@ref).
"""
function sphere_descriptor_pixels(descs::AbstractVector{<:SphereDescriptor},
                                  Npe::Int, Nfe::Int, FOV::Real)
    [sphere_descriptor_pixel(d, Npe, Nfe, FOV) for d in descs]
end

function _empty_phantom(name::AbstractString = "empty")
    Phantom(name = String(name), x = Float64[])
end

function _slice_plane_basis(cfg::PhantomConfig)
    n_hat, u_hat, v_hat = slice_basis(cfg.slice_normal)
    centre = _mm_to_m.(cfg.slice_center_mm)
    centre, n_hat, u_hat, v_hat
end

function _slice_slab(cfg::PhantomConfig)
    cfg.slice_thickness_mm === nothing && return nothing
    centre, n_hat, _, _ = _slice_plane_basis(cfg)
    Slab(centre, n_hat, _mm_to_m(cfg.slice_thickness_mm) / 2)
end

function _water_voxel_size_mm(cfg::PhantomConfig)
    cfg.water_voxel_size_mm === nothing ? cfg.voxel_size_mm : cfg.water_voxel_size_mm
end

function _water_throughplane_voxel_size_mm(cfg::PhantomConfig)
    cfg.water_throughplane_voxel_size_mm === nothing ?
        something(cfg.slice_thickness_mm, cfg.voxel_size_mm) :
        cfg.water_throughplane_voxel_size_mm
end

function _water_ρ_weight(cfg::PhantomConfig)
    sphere_dx = _mm_to_m(cfg.voxel_size_mm)
    water_dx = _mm_to_m(_water_voxel_size_mm(cfg))
    if cfg.slice_thickness_mm !== nothing
        through_dx = _mm_to_m(_water_throughplane_voxel_size_mm(cfg))
        return (water_dx / sphere_dx)^2 * (through_dx / sphere_dx)
    end
    cfg.water_voxel_size_mm === nothing && return 1.0
    (water_dx / sphere_dx)^3
end

function _slice_sheet_offsets_and_thicknesses(slice_thickness_m::Real,
                                              sheet_spacing_m::Real)
    T = Float64(slice_thickness_m)
    dz = Float64(sheet_spacing_m)
    T > 0 || error("slice_thickness_mm must be > 0")
    dz > 0 || error("water_throughplane_voxel_size_mm must be > 0")

    edges = collect(-T / 2 : dz : T / 2)
    if isempty(edges) || edges[1] > -T / 2 + PLANE_TOL
        pushfirst!(edges, -T / 2)
    end
    if edges[end] < T / 2 - PLANE_TOL
        push!(edges, T / 2)
    else
        edges[end] = T / 2
    end

    offsets = Float64[]
    thicknesses = Float64[]
    for i in 1:(length(edges) - 1)
        thickness = edges[i + 1] - edges[i]
        thickness > PLANE_TOL || continue
        push!(offsets, (edges[i] + edges[i + 1]) / 2)
        push!(thicknesses, thickness)
    end
    offsets, thicknesses
end

"""
    build_phantom_from_descriptors(descs, delta_x; name, slab) -> Phantom

Voxelise a vector of [`SphereDescriptor`](@ref)s at grid spacing `delta_x`
(metres) and concatenate into a single `KomaMRI.Phantom`. Returns an empty
phantom if `descs` is empty or all spheres miss the slab.
"""
function build_phantom_from_descriptors(descs::AbstractVector{<:SphereDescriptor},
                                        delta_x::Real;
                                        name::AbstractString = "spheres",
                                        slab::Union{Nothing,Slab} = nothing)
    isempty(descs) && return _empty_phantom(name)
    parts = Phantom[]
    for d in descs
        p = build_sphere(d, delta_x; slab = slab)
        length(p.x) > 0 && push!(parts, p)
    end
    isempty(parts) && return _empty_phantom(name)
    obj = reduce(+, parts)
    obj.name = String(name)
    obj
end

"""
    build_sphere(d, delta_x; name, slab) -> Phantom

Voxelise a single [`SphereDescriptor`](@ref) at grid spacing `delta_x` (metres).
Returns an empty phantom if no grid points fall inside the sphere (or slab).
"""
function build_sphere(d::SphereDescriptor, delta_x::Real;
                      name::AbstractString = String(d.label),
                      slab::Union{Nothing,Slab} = nothing)
    x, y, z = voxelise_sphere(d.centre, d.radius, delta_x; slab = slab)
    n = length(x)
    n == 0 && return _empty_phantom(name)
    Phantom(
        name = String(name),
        x = x, y = y, z = z,
        ρ   = fill(d.ρ,   n),
        T1  = fill(d.T1,  n),
        T2  = fill(d.T2,  n),
        T2s = fill(d.T2s, n),
        Δw  = fill(d.delta_w, n),
    )
end

"""
    build_plate(plate, cfg) -> Phantom

Voxelise the sphere descriptors of one plate and concatenate into a single
`Phantom`. Returns an empty phantom if the plate is empty after augmentation.
"""
function build_plate(plate::Symbol, cfg::PhantomConfig)
    rng = Random.MersenneTwister(cfg.rng_seed + hash(plate) % PLATE_RNG_BUCKETS)
    descs = sphere_descriptors(plate, cfg; rng)
    delta_x = _mm_to_m(cfg.voxel_size_mm)
    build_phantom_from_descriptors(descs, delta_x; name = String(plate))
end

"""
    build_background_water(cfg; cutout_descs=nothing) -> Phantom

Voxelise a 100 mm-radius water sphere and cut out all contrast + fiducial
sphere volumes. Spins are labelled as bulk water.
"""
function build_background_water(cfg::PhantomConfig; cutout_descs = nothing)
    water, _ = _build_background_water_with_weights(cfg; cutout_descs = cutout_descs)
    water
end

function _build_background_water_with_weights(cfg::PhantomConfig; cutout_descs = nothing)
    sphere_dx = _mm_to_m(cfg.voxel_size_mm)
    sphere_dx > 0 || error("voxel_size_mm must be > 0")
    water_dx = _mm_to_m(_water_voxel_size_mm(cfg))
    water_dx > 0 || error("water_voxel_size_mm must be > 0")
    if cfg.water_throughplane_voxel_size_mm !== nothing
        cfg.water_throughplane_voxel_size_mm > 0 ||
            error("water_throughplane_voxel_size_mm must be > 0")
    end
    props = BACKGROUND_WATER[cfg.field]

    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    ρ_weight = Float64[]

    if cfg.slice_thickness_mm !== nothing
        centre, n_hat, u_hat, v_hat = _slice_plane_basis(cfg)
        offsets, thicknesses = _slice_sheet_offsets_and_thicknesses(
            _mm_to_m(cfg.slice_thickness_mm),
            _mm_to_m(_water_throughplane_voxel_size_mm(cfg)))
        area_weight = (water_dx / sphere_dx)^2
        for (offset, thickness) in zip(offsets, thicknesses)
            sheet_centre = (
                centre[1] + offset * n_hat[1],
                centre[2] + offset * n_hat[2],
                centre[3] + offset * n_hat[3],
            )
            x, y, z = voxelise_plane(sheet_centre, n_hat, u_hat, v_hat,
                                     water_dx, HOUSING_RADIUS_M)
            append!(xs, x)
            append!(ys, y)
            append!(zs, z)
            append!(ρ_weight, fill(area_weight * (thickness / sphere_dx), length(x)))
        end
    else
        xs, ys, zs = voxelise_sphere((0.0, 0.0, 0.0), HOUSING_RADIUS_M, water_dx)
        append!(ρ_weight, fill(_water_ρ_weight(cfg), length(xs)))
    end
    isempty(xs) && return _empty_phantom("water"), Float64[]
    # Keep sliced water under the same tolerant slab mask used for sphere voxels.
    slab = _slice_slab(cfg)
    if slab !== nothing
        slice_ok = _slice_slab_mask(xs, ys, zs, slab)
        xs, ys, zs = xs[slice_ok], ys[slice_ok], zs[slice_ok]
        ρ_weight = ρ_weight[slice_ok]
    end
    isempty(xs) && return _empty_phantom("water"), Float64[]
    keep = trues(length(xs))
    cutouts = cutout_descs === nothing ? all_sphere_descriptors(cfg) : cutout_descs
    for d in cutouts
        @. keep &= ((xs - d.centre[1])^2 + (ys - d.centre[2])^2 +
                    (zs - d.centre[3])^2) > d.radius^2
    end
    sum(keep) == 0 && return _empty_phantom("water"), Float64[]
    n = sum(keep)
    ρ_weight = ρ_weight[keep]
    water = Phantom(
        name = "water",
        x = xs[keep], y = ys[keep], z = zs[keep],
        ρ   = props.ρ .* ρ_weight,
        T1  = fill(props.T1, n),
        T2  = fill(props.T2, n),
        T2s = fill(props.T2, n),
        Δw  = fill(0.0,      n),
    )
    water, ρ_weight
end

"""
    build_phantom(cfg) -> Phantom

Assemble the full QalibreMD Model 130 digital twin from a `PhantomConfig`.
Pipeline: descriptors → voxelise → background water → rotate/translate →
per-spin noise. The returned `Phantom` can be fed directly to
`KomaMRI.simulate`.
"""
function build_phantom(cfg::PhantomConfig = PhantomConfig())
    rng = Random.MersenneTwister(cfg.rng_seed)
    sphere_descs = all_sphere_descriptors(cfg; rng)
    slab = _slice_slab(cfg)
    parts = Phantom[]
    ρ_weight_parts = Vector{Vector{Float64}}()
    if !isempty(sphere_descs)
        spheres = build_phantom_from_descriptors(
            sphere_descs, _mm_to_m(cfg.voxel_size_mm); name = "spheres", slab = slab)
        if length(spheres.x) > 0
            push!(parts, spheres)
            push!(ρ_weight_parts, ones(length(spheres.x)))
        end
    end
    if :water ∈ cfg.include_plates
        water, water_ρ_weight = _build_background_water_with_weights(
            cfg; cutout_descs = sphere_descs)
        if length(water.x) > 0
            push!(parts, water)
            push!(ρ_weight_parts, water_ρ_weight)
        end
    end

    obj = isempty(parts) ? _empty_phantom("qalibremd") : reduce(+, parts)
    ρ_weight = isempty(ρ_weight_parts) ? Float64[] : reduce(vcat, ρ_weight_parts)
    obj.name = "qalibremd"
    if cfg.slice_thickness_mm !== nothing && length(obj.x) > 0
        # Tolerance keeps voxels that land exactly on the slab edge despite
        # floating-point roundoff.
        keep = _slice_slab_mask(obj.x, obj.y, obj.z, slab)
        obj = obj[keep]
        ρ_weight = ρ_weight[keep]
        obj.name = "qalibremd"
    end
    obj = apply_transform!(obj, cfg.rotation, _mm_to_m.(cfg.translation_mm))
    obj = apply_per_spin_noise!(obj, cfg.augment, rng; ρ_weight = ρ_weight)
    obj
end
