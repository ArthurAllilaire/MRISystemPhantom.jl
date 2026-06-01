"""
    _slab_axis_bands(centre, radius, slab)

Return conservative x/y/z coordinate bands for the part of a sphere bounding
box that can intersect an oriented slab. These bands are only a cheap
pre-filter; the exact signed-distance slab mask is still applied after the
full lattice is built.
"""
function _slab_axis_bands(centre, radius, slab::Slab)
    c = _vec3(centre)
    p = _vec3(slab.centre)
    n = _vec3(slab.n_hat)
    r = Float64(radius)
    τ = slab.half_thickness

    function axis_band(axis)
        ni = n[axis]

        # If the slab normal is almost perpendicular to this coordinate axis,
        # moving along this axis does not usefully change signed distance to
        # the slab, so keep the full sphere bounding-box extent.
        abs(ni) <= PLANE_TOL && return c[axis] - r, c[axis] + r

        # Split the slab equation into the selected axis contribution plus the
        # two remaining axes. The remaining axes can each vary by +/-r within
        # the sphere's bounding box, giving a conservative contribution margin.
        other_axes = filter(!=(axis), (1, 2, 3))
        b0 = sum(n[j] * (c[j] - p[j]) for j in other_axes)
        bmargin = r * sum(abs(n[j]) for j in other_axes)

        # Solve for the selected axis contribution that could still satisfy
        # -τ <= dot(n, x - p) <= τ after the other axes take any in-box value.
        lo_proj = -τ - (b0 + bmargin)
        hi_proj =  τ - (b0 - bmargin)

        # Divide by the selected normal component. Negative components reverse
        # the interval ordering, so swap the projected limits in that case.
        lo_axis, hi_axis = ni > 0 ?
            (p[axis] + lo_proj / ni, p[axis] + hi_proj / ni) :
            (p[axis] + hi_proj / ni, p[axis] + lo_proj / ni)

        # The slab-derived band is intersected with the sphere bounding box.
        max(c[axis] - r, lo_axis), min(c[axis] + r, hi_axis)
    end

    axis_band(1), axis_band(2), axis_band(3)
end

"""
    voxelise_sphere(centre, radius, delta_x; slab=nothing) -> (xs, ys, zs)

Return voxel-centre coordinates (in metres, `Vector{Float64}` each) that lie
inside a sphere of given `centre` (m) and `radius` (m), sampled on an
isotropic cubic lattice of spacing `delta_x` (m). The lattice for each sphere is
locally aligned to its centre; points from different spheres are not
guaranteed to share a global grid, but that is fine for a bag-of-spins
phantom. When provided, `slab` clips the sphere to an oriented `Slab`.
"""
function voxelise_sphere(centre::NTuple{3,<:Real}, radius::Real, delta_x::Real;
                         slab::Union{Nothing,Slab} = nothing)
    cx, cy, cz = Float64.(centre)
    r = Float64(radius)
    h = Float64(delta_x)
    xs = cx - r : h : cx + r
    ys = cy - r : h : cy + r
    zs = cz - r : h : cz + r
    # Optional oriented slab restriction. First shrink to a conservative
    # axis-aligned bounding box to avoid materialising obvious out-of-slab
    # coordinates, then apply the exact signed-distance mask below.
    if slab !== nothing
        (xlo, xhi), (ylo, yhi), (zlo, zhi) =
            _slab_axis_bands((cx, cy, cz), r, slab)
        # Keep the original centre-aligned lattice, just drop coordinate values
        # whose whole grid planes cannot intersect the slab.
        xs = filter(x -> xlo - SLAB_BOUNDARY_TOL <= x <= xhi + SLAB_BOUNDARY_TOL,
                    collect(xs))
        ys = filter(y -> ylo - SLAB_BOUNDARY_TOL <= y <= yhi + SLAB_BOUNDARY_TOL,
                    collect(ys))
        zs = filter(z -> zlo - SLAB_BOUNDARY_TOL <= z <= zhi + SLAB_BOUNDARY_TOL,
                    collect(zs))
    end
    X = [x for x in xs, _ in ys, _ in zs]
    Y = [y for _ in xs, y in ys, _ in zs]
    Z = [z for _ in xs, _ in ys, z in zs]
    mask = @. (X - cx)^2 + (Y - cy)^2 + (Z - cz)^2 <= r^2
    if slab !== nothing
        mask .&= _slice_slab_mask(X, Y, Z, slab)
    end
    return X[mask], Y[mask], Z[mask]
end

"""
    sphere_volume(radius)

Analytic sphere volume (m³) — used by tests as a ground truth for the
voxelised count.
"""
sphere_volume(radius::Real) = (4/3) * π * radius^3
