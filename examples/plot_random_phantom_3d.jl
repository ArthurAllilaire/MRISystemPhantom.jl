# Interactive 3D render of a randomly-arranged "spheres-in-water" phantom, using
# the reusable `plot_phantom_html` viewer. The random spheres go in through the
# `custom_sphere_descriptors` path and become their own scatter trace; the
# background water is a separate translucent trace with an opacity slider, and a
# dropdown switches the sphere colouring between T1/T2/T2s/ρ/Δw.
#
# For the full QalibreMD Model 130 phantom (per-plate traces), see
# examples/plot_phantom_3d.jl.
#
# Run with:  julia --project=. examples/plot_random_phantom_3d.jl

using MRISystemPhantom
using KomaMRI   # loads PlotlyJS transitively, which activates the viewer extension
using Random

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")
isdir(ASSETS) || mkpath(ASSETS)

# ---------- random sphere generation ------------------------------------
"""
    random_sphere_descriptors(rng; n, region_mm, radius_mm)

Sample `n` non-overlapping `SphereDescriptor`s with random centres, radii and
relaxation times. Centres stay within a sphere of radius `region_mm` of the
origin (well inside the 100 mm water housing) so water fully surrounds them.
"""
function random_sphere_descriptors(rng::AbstractRNG;
        n::Int = 12, region_mm::Real = 60.0,
        radius_mm::Tuple{<:Real,<:Real} = (6.0, 14.0))
    descs = SphereDescriptor[]
    attempts = 0
    while length(descs) < n && attempts < 10_000
        attempts += 1
        r    = (radius_mm[1] + rand(rng) * (radius_mm[2] - radius_mm[1])) * 1e-3
        # Sample a centre uniformly inside the working sphere (reject-sample the cube).
        c_mm = ntuple(_ -> (2rand(rng) - 1) * region_mm, 3)
        sqrt(sum(abs2, c_mm)) + r * 1e3 > region_mm && continue   # keep inside region
        centre = c_mm .* 1e-3
        any(d -> sqrt(sum(abs2, centre .- d.centre)) < r + d.radius, descs) && continue
        # Random but physically-ordered relaxation: T2 < T1.
        T1  = 0.2 + rand(rng) * 1.8                # 0.2 – 2.0 s
        T2  = T1 * (0.1 + rand(rng) * 0.3)         # 10–40 % of T1
        push!(descs, SphereDescriptor(centre, r, 1.0, T1, T2, T2, 0.0,
                                      Symbol("sphere_$(length(descs) + 1)")))
    end
    length(descs) < n && @warn "Only placed $(length(descs))/$n spheres" attempts
    descs
end

# ---------- build a random phantom and render ----------------------------
rng   = MersenneTwister(20240608)
descs = random_sphere_descriptors(rng; n = 12)

cfg = PhantomConfig(
    field          = :T15,
    voxel_size_mm  = 2.0,
    include_plates = [:water],            # only background water + our custom spheres
    water_voxel_size_mm = 3.0,            # coarser water → fewer spins to render
    custom_sphere_descriptors = descs,
)

out = joinpath(ASSETS, "random_phantom_interactive_3d.html")
plot_phantom_html(cfg; color_by = :T1, file = out)
@info "Saved random-phantom render" spheres = length(descs) file = out
