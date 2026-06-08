# Interactive 3D phantom viewer — the reusable `plot_phantom_html` interface.
#
# `plot_phantom_html(cfg; ...)` renders any phantom as one Plotly scatter3d trace
# per group (T1 / T2 / PD plates, fiducials, custom spheres, water). Each group is
# show/hideable from the legend, the spheres are coloured by a property you can
# switch live (T1 / T2 / T2s / ρ / Δw dropdown), and the translucent water has an
# opacity slider. Pass `color_water = true` if water should use the same property
# colour axis instead of a constant light-blue trace. It lives in a PlotlyJS package
# *extension* (no hard plotting dependency on the core package); the extension
# activates as soon as PlotlyJS is in the session, which `using KomaMRI` pulls in
# transitively.
#
# Run with:  julia --project=. examples/plot_phantom_3d.jl

using MRISystemPhantom
using KomaMRI   # loads PlotlyJS transitively, which activates the viewer extension

const ASSETS = joinpath(@__DIR__, "..", "src", "assets")
isdir(ASSETS) || mkpath(ASSETS)

# Full QalibreMD Model 130 phantom — one trace per plate (T1/T2/PD/fiducials)
# plus translucent water. For a randomly-arranged spheres-in-water phantom, see
# examples/plot_random_phantom_3d.jl.
cfg = PhantomConfig(field = :T3, voxel_size_mm = 1.5,
                    water_voxel_size_mm = 3.0)   # coarser water → fewer spins

out = joinpath(ASSETS, "phantom_interactive_3d.html")
plot_phantom_html(cfg; color_by = :T1, file = out)
@info "Saved full-phantom render" file = out
