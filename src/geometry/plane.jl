const PLANE_TOL = 1e-12
const SLAB_BOUNDARY_TOL = 1e-9

_tuple3(v) = (Float64(v[1]), Float64(v[2]), Float64(v[3]))
_vec3(v) = Float64[v[1], v[2], v[3]]

"""
    Slab(centre, normal, half_thickness)

An oriented slab in metres. `centre` is a point on the centre plane, `n_hat`
is the unit normal, and `half_thickness` is the distance from the centre plane
to either slab face.
"""
struct Slab
    centre::NTuple{3,Float64}
    n_hat::NTuple{3,Float64}
    half_thickness::Float64

    function Slab(centre::NTuple{3,<:Real},
                  normal::NTuple{3,<:Real},
                  half_thickness::Real)
        n = _vec3(normal)
        n_norm = norm(n)
        n_norm > PLANE_TOL || error("slab normal must have nonzero norm")
        τ = Float64(half_thickness)
        τ >= 0 || error("slab half_thickness must be >= 0")

        new(_tuple3(centre), _tuple3(n ./ n_norm), τ)
    end
end

"""
    slice_basis(normal) -> (n_hat, u_hat, v_hat)

Build an orthonormal basis for a slice plane. `n_hat` is the normalised
normal; `u_hat` and `v_hat` span the plane.
"""
function slice_basis(normal::NTuple{3,<:Real})
    n = _vec3(normal)
    n_norm = norm(n)
    n_norm > PLANE_TOL || error("slice_normal must have nonzero norm")
    n ./= n_norm

    axes = (
        Float64[1.0, 0.0, 0.0],
        Float64[0.0, 1.0, 0.0],
        Float64[0.0, 0.0, 1.0],
    )
    axis = axes[argmin(abs.(n))]
    u = cross(axis, n)
    u ./= norm(u)
    v = cross(n, u)
    v ./= norm(v)

    _tuple3(n), _tuple3(u), _tuple3(v)
end

"""
    signed_distance(x, y, z, centre, n_hat)

Signed distance from point `(x, y, z)` to the plane through `centre` with unit
normal `n_hat`. dot(point-centre, n)
"""
function signed_distance(x::Real, y::Real, z::Real,
                         centre::NTuple{3,<:Real},
                         n_hat::NTuple{3,<:Real})
    (Float64(x) - centre[1]) * n_hat[1] +
    (Float64(y) - centre[2]) * n_hat[2] +
    (Float64(z) - centre[3]) * n_hat[3]
end

function _slice_slab_mask(xs, ys, zs, centre, n_hat, half_thickness_m)
    abs.(signed_distance.(xs, ys, zs, Ref(centre), Ref(n_hat))) .<=
        half_thickness_m + SLAB_BOUNDARY_TOL
end

_slice_slab_mask(xs, ys, zs, slab::Slab) =
    _slice_slab_mask(xs, ys, zs, slab.centre, slab.n_hat, slab.half_thickness)

"""
    voxelise_plane(centre, n_hat, u_hat, v_hat, in_plane_dx, radius)

Return metre coordinates on a 2D lattice lying in the plane and clipped to the
sphere of radius `radius` centred at the origin.
"""
function voxelise_plane(centre::NTuple{3,<:Real},
                        n_hat::NTuple{3,<:Real},
                        u_hat::NTuple{3,<:Real},
                        v_hat::NTuple{3,<:Real},
                        in_plane_dx::Real,
                        radius::Real)
    h = Float64(in_plane_dx)
    h > 0 || error("in_plane_dx must be > 0")
    R = Float64(radius)
    R >= 0 || error("radius must be >= 0")

    c = _vec3(centre)
    n = _vec3(n_hat)
    u = _vec3(u_hat)
    v = _vec3(v_hat)
    d = dot(c, n)
    abs(d) <= R + PLANE_TOL || return Float64[], Float64[], Float64[]

    r_plane = sqrt(max(0.0, R^2 - d^2))
    imax = ceil(Int, r_plane / h)
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    for i in -imax:imax, j in -imax:imax
        p = c .+ (i * h) .* u .+ (j * h) .* v
        if dot(p, p) <= R^2 + PLANE_TOL
            push!(xs, p[1])
            push!(ys, p[2])
            push!(zs, p[3])
        end
    end

    xs, ys, zs
end
