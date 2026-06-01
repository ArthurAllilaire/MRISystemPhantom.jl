"""
    contrast_plate_centres(z_mm; …) -> Vector{NTuple{3,Float64}}

14 contrast-sphere centres (metres) for a plate at depth `z_mm`, arranged as
two concentric rings — outer 10 and inner 4 — matching the manual photos.
Ring radii and starting phases can be overridden.
"""
function contrast_plate_centres(z_mm::Real;
                                n_outer::Int = 10, n_inner::Int = 4,
                                r_outer_mm::Real = 65.0, r_inner_mm::Real = 28.0,
                                phase0_outer::Real = 0.0,
                                phase0_inner::Real = π/4)
    centres = NTuple{3,Float64}[]
    z = Float64(z_mm) * 1e-3
    for k in 0:n_outer-1
        θ = phase0_outer + 2π * k / n_outer
        push!(centres, (r_outer_mm * cos(θ) * 1e-3,
                        r_outer_mm * sin(θ) * 1e-3,
                        z))
    end
    for k in 0:n_inner-1
        θ = phase0_inner + 2π * k / n_inner
        push!(centres, (r_inner_mm * cos(θ) * 1e-3,
                        r_inner_mm * sin(θ) * 1e-3,
                        z))
    end
    centres
end

"""
    fiducial_grid_centres(; spacing_mm = 40.0, rmax_mm = 95.0)

Cubic grid of fiducial-sphere centres (metres), clipped to a sphere of
radius `rmax_mm`. With the default parameters this produces 57 points,
matching the manual.
"""
function fiducial_grid_centres(; spacing_mm::Real = 40.0,
                                 rmax_mm::Real = 95.0)
    pts = NTuple{3,Float64}[]
    ns = -2:2
    for i in ns, j in ns, k in ns
        p_mm = (i * spacing_mm, j * spacing_mm, k * spacing_mm)
        if p_mm[1]^2 + p_mm[2]^2 + p_mm[3]^2 <= rmax_mm^2
            push!(pts, p_mm .* 1e-3)
        end
    end
    pts
end

"Plate z-positions in mm (manual §2.2)."
const PLATE_Z_MM = (T1 = 56.5, T2 = 16.5, PD = -23.5)
