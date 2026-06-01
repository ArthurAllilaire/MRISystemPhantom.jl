# PD-array spheres — H₂O / D₂O mixtures (PD-1 … PD-14)
# PD value == volume fraction of H₂O.

"Proton density fractions (H₂O volume fraction) for the 14 PD-plate spheres. Range 0.05–1.00."
const PD_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# First-order approximation: T1/T2 of H₂O/D₂O mixtures ≈ bulk water values.
# Exposed as callables so future calibrations can swap in a ρ-dependent model
# without touching the builder.
pd_t1(ρ::Real, field::Symbol) = BACKGROUND_WATER[field].T1
pd_t2(ρ::Real, field::Symbol) = BACKGROUND_WATER[field].T2
