"""
    rotation_matrix(α, β, γ)

Intrinsic XYZ Euler rotation (radians): Rx(α) * Ry(β) * Rz(γ) applied as
`R * v`. Matches the convention used by `PhantomConfig.rotation`.
"""
function rotation_matrix(α::Real, β::Real, γ::Real)
    Rx = [1 0 0; 0 cos(α) -sin(α); 0 sin(α) cos(α)]
    Ry = [cos(β) 0 sin(β); 0 1 0; -sin(β) 0 cos(β)]
    Rz = [cos(γ) -sin(γ) 0; sin(γ) cos(γ) 0; 0 0 1]
    Rx * Ry * Rz
end

"""
    apply_transform!(obj, euler, translation)

Rotate every spin position of `obj` by the given Euler angles (radians)
and then translate by `translation` (metres). Returns the same phantom
(positions are mutated in place).
"""
function apply_transform!(obj, euler::NTuple{3,<:Real}, translation::NTuple{3,<:Real})
    length(obj.x) == 0 && return obj
    R = rotation_matrix(euler...)
    @inbounds for i in eachindex(obj.x)
        v = R * [obj.x[i], obj.y[i], obj.z[i]]
        obj.x[i] = v[1] + translation[1]
        obj.y[i] = v[2] + translation[2]
        obj.z[i] = v[3] + translation[3]
    end
    obj
end

"""
    apply_per_spin_noise!(obj, aug, rng; ρ_weight=nothing)

Gaussian jitter on T1, T2, ρ, positions, and off-resonance. Sigmas of zero
are no-ops so by default this function is effectively a passthrough.
When `ρ_weight` is provided, PD jitter is applied to material proton density
`obj.ρ[i] / ρ_weight[i]`, clamped to `[0, 1]`, then multiplied by the weight.
"""
function apply_per_spin_noise!(obj, aug::AugmentConfig, rng::AbstractRNG;
                               ρ_weight::Union{Nothing,AbstractVector{<:Real}} = nothing)
    n = length(obj.x)
    n == 0 && return obj
    if ρ_weight !== nothing && length(ρ_weight) != n
        error("ρ_weight length must match phantom spin count")
    end

    if aug.T1_sigma_rel > 0
        @inbounds for i in 1:n
            obj.T1[i] = max(1e-6, obj.T1[i] * (1 + aug.T1_sigma_rel * randn(rng)))
        end
    end
    if aug.T2_sigma_rel > 0
        @inbounds for i in 1:n
            t2 = max(1e-6, obj.T2[i] * (1 + aug.T2_sigma_rel * randn(rng)))
            obj.T2[i] = t2
            obj.T2s[i] = min(obj.T2s[i], t2)
        end
    end
    if aug.PD_sigma_abs > 0
        @inbounds for i in 1:n
            if ρ_weight === nothing
                obj.ρ[i] = clamp(obj.ρ[i] + aug.PD_sigma_abs * randn(rng), 0.0, 1.0)
            else
                w = Float64(ρ_weight[i])
                w > 0 || error("ρ_weight entries must be > 0")
                ρ_material = obj.ρ[i] / w
                ρ_material = clamp(ρ_material + aug.PD_sigma_abs * randn(rng), 0.0, 1.0)
                obj.ρ[i] = ρ_material * w
            end
        end
    end
    if aug.position_sigma_mm > 0
        σ = aug.position_sigma_mm * 1e-3
        @inbounds for i in 1:n
            obj.x[i] += σ * randn(rng)
            obj.y[i] += σ * randn(rng)
            obj.z[i] += σ * randn(rng)
        end
    end
    if aug.B0_sigma_Hz > 0
        omega = 2π * aug.B0_sigma_Hz
        @inbounds for i in 1:n
            obj.Δw[i] += omega * randn(rng)
        end
    end

    obj
end
