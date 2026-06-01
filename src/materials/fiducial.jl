# Fiducial spheres — aqueous CuSO₄.
# The manual excerpt does not tabulate T1/T2 for the fiducials. The values
# below are generic CuSO₄ placeholders, NOT verified against a specific
# calibration record — replace them when real numbers land. Candidate
# sources for traceable per-sphere calibration:
#
#   • MRIStandards/SystemPhantom repo (carries calibration PDFs for a
#     related NIST/ISMRM phantom):
#       https://github.com/MRIStandards/SystemPhantom
#   • NIST Magnetic Imaging Group (datasets + phantom papers):
#       https://www.nist.gov/pml/applied-physics-division/magnetic-imaging
#   • NIST publication — comparison of T1 measurement using the
#     ISMRM/NIST system phantom:
#       https://www.nist.gov/publications/comparison-t1-measurement-using-ismrmnist-system-phantom
#   • ISMRM 2012 abstract #2456 / NIST MDS2-2366 dataset
#     (original characterisation).

struct Relax
    T1::Float64   # s
    T2::Float64   # s
    ρ::Float64
end

"Relaxation properties of the fiducial spheres (aqueous CuSO₄). **Placeholder values** — not verified against calibration records. See file header for traceable sources."
const FIDUCIAL_PROPS = Dict(
    :T15 => Relax(0.180, 0.120, 1.0),
    :T3  => Relax(0.180, 0.120, 1.0),
)
