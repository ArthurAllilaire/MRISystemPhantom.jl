# Background — deionised / distilled water filling the hemispheres.
# T1/T2 at 20 °C, SI units.

"Relaxation properties of the background water housing at 1.5 T and 3 T (distilled water at 20 °C)."
const BACKGROUND_WATER = Dict(
    :T15 => Relax(2.8, 2.2, 1.0),
    :T3  => Relax(3.0, 2.0, 1.0),
)
