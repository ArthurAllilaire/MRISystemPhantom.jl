"T1 values (s) for the 14 NiCl₂ T1-plate spheres at 1.5 T and 3 T. Serial class ≥ 0042. Source: QalibreMD Model 130 manual."
const T1_ARRAY = Dict(
    :T15 => [1.879, 1.432, 1.027, 0.7513, 0.527, 0.3841, 0.2723, 0.1945,
             0.1378, 0.0947, 0.067, 0.04814, 0.03435, 0.02416],
    :T3  => [1.838, 1.398, 0.9983, 0.7258, 0.5091, 0.367, 0.2587, 0.1847,
             0.1308, 0.0909, 0.0642, 0.04628, 0.03265, 0.02295],
)

"T2 companion values (s) for each T1-plate sphere at 1.5 T and 3 T. Paired with [`T1_ARRAY`](@ref)."
const T2_OF_T1_ARRAY = Dict(
    :T15 => [1.542, 1.196, 0.8717, 0.6461, 0.4575, 0.3353, 0.2384, 0.1706,
             0.1216, 0.0837, 0.0592, 0.0426, 0.0304, 0.0213],
    :T3  => [1.354, 1.035, 0.7283, 0.5244, 0.3686, 0.2667, 0.1893, 0.1341,
             0.0938, 0.0657, 0.0468, 0.03315, 0.02369, 0.01673],
)

"T1 values (s) for serial class 0001–0041 (legacy recipe). Substantially different from [`T1_ARRAY`](@ref) — use the correct class for your phantom serial number."
const T1_ARRAY_LEGACY = Dict(
    :T15 => [2.480, 2.173, 1.907, 1.604, 1.332, 1.044, 0.8014, 0.6089,
             0.4538, 0.3382, 0.2514, 0.1886, 0.1407, 0.1054],
    :T3  => [2.376, 2.067, 1.797, 1.504, 1.244, 0.9733, 0.7459, 0.5666,
             0.4221, 0.3145, 0.2339, 0.1751, 0.1305, 0.0974],
)
