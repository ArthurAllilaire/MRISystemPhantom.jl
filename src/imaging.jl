# MRI imaging-pipeline utilities: k-space ↔ image reconstruction and noise.
# These are shared between the RL environments (e2.jl) and diagnostics (snr.jl).

using FFTW

"""
    add_noise!(ksp, σ; rng = Random.GLOBAL_RNG) → ksp

Add absolute complex Gaussian noise to a k-space matrix in place. Real and
imaginary parts are independently `N(0, σ²)` (Julia convention for
`randn(ComplexF32, …)`). This is the physically-correct model (FIX_SIM_PLAN §2):
MRI thermal noise is hardware-determined and scene-independent, and after
`abs()` the residuals are Rician — which is what we want at low SNR. No-op
when `σ ≤ 0`. Use this everywhere noise is added to k-space.
"""
function add_noise!(ksp::AbstractMatrix{<:Complex}, σ::Real;
                    rng::AbstractRNG = Random.GLOBAL_RNG)
    σ > 0 || return ksp
    ksp .+= Float32(σ) .* randn(rng, ComplexF32, size(ksp))
    ksp
end

"""
    add_noise(ksp, σ; rng = Random.GLOBAL_RNG) → new ksp

Allocating, non-mutating counterpart to [`add_noise!`](@ref). Returns a
fresh array; the input is untouched. Equivalent to
`add_noise!(copy(ksp), σ; rng = rng)`. Convenient when adding multiple
independent noise realisations to the same clean k-space.
"""
add_noise(ksp::AbstractMatrix{<:Complex}, σ::Real;
          rng::AbstractRNG = Random.GLOBAL_RNG) =
    add_noise!(copy(ksp), σ; rng = rng)

"""
    add_gaussian_noise!(x, σ; rng = Random.GLOBAL_RNG) → x

Add real Gaussian noise in place. Kept as a reference implementation; not used
anywhere in the codebase (E2 noise is complex via [`add_noise!`](@ref)). Do not
call from new code unless you specifically want real-only Gaussian noise.
"""
function add_gaussian_noise!(x::AbstractArray{<:Real}, σ::Real;
                             rng::AbstractRNG = Random.GLOBAL_RNG)
    σ > 0 || return x
    x .+= Float32(σ) .* randn(rng, Float32, size(x))
    x
end

"""
    roi_mean(img, ipe, ife; r=0) → Float64

Mean over a `(2r+1)²` square ROI centred on `(ipe, ife)`, clamped to image
bounds. `r=0` returns the single centre pixel as a `Float64`. Used to extract
per-sphere magnitude values from a reconstructed image.
"""
@inline function roi_mean(img::AbstractMatrix{<:Real}, ipe::Int, ife::Int; r::Int = 0)
    Npe, Nfe = size(img)
    pe_lo = clamp(ipe - r, 1, Npe)
    pe_hi = clamp(ipe + r, 1, Npe)
    fe_lo = clamp(ife - r, 1, Nfe)
    fe_hi = clamp(ife + r, 1, Nfe)
    s = 0.0
    n = 0
    @inbounds for j in fe_lo:fe_hi, i in pe_lo:pe_hi
        s += Float64(img[i, j])
        n += 1
    end
    s / n
end

"""
    raw_to_kspace(raw, Npe, Nfe) → Matrix{ComplexF32}

Extract the Npe×Nfe k-space matrix from KomaMRI raw simulation output.
"""
function raw_to_kspace(raw, Npe::Int, Nfe::Int)
    ksp = zeros(ComplexF32, Npe, Nfe)
    for k in 1:Npe
        k <= length(raw.profiles) || continue
        ksp[k, :] = ComplexF32.(raw.profiles[k].data[:, 1])
    end
    ksp
end

"""
    hamming_window_2d(Npe, Nfe) → Matrix{Float32}

Separable 2D Hamming window of size `Npe × Nfe`. Multiply k-space by this
before IFFT to suppress sidelobes from −13 dB to −43 dB, at the cost of a
~3× DC attenuation and a wider main lobe. Exposed so callers can dump the
windowed k-space directly (rather than passing `hamming=true` to
`kspace_to_image`, which only returns the image).
"""
function hamming_window_2d(Npe::Integer, Nfe::Integer)
    w_pe = Float32.(0.54 .- 0.46 .* cos.(2π .* (0:Npe-1) ./ (Npe-1)))
    w_fe = Float32.(0.54 .- 0.46 .* cos.(2π .* (0:Nfe-1) ./ (Nfe-1)))
    w_pe .* transpose(w_fe)
end

"""
    kspace_to_image(ksp; phase_sensitive=false, pad_factor=1, hamming=false) → Matrix{Float32}

2D IFFT reconstruction: ifftshift → ifft → fftshift.
Returns magnitude image by default; signed real part when `phase_sensitive=true`.

Optional clinical recon steps:
* `hamming=true`  — apply a 2D Hamming window (sidelobes −13 dB → −43 dB) before IFFT.
* `pad_factor>1`  — symmetric zero-padding before IFFT (image interpolation only,
  no info added).
"""
function kspace_to_image(ksp::Matrix{ComplexF32};
                         phase_sensitive::Bool=false,
                         pad_factor::Int=1,
                         hamming::Bool=false)
    ksp_w = hamming ? ksp .* hamming_window_2d(size(ksp)...) : ksp
    if pad_factor > 1
        Npe_k, Nfe_k = size(ksp_w)
        Npe_pad = Npe_k * pad_factor
        Nfe_pad = Nfe_k * pad_factor
        ksp_padded = zeros(ComplexF32, Npe_pad, Nfe_pad)
        pe0 = (Npe_pad - Npe_k) ÷ 2 + 1
        fe0 = (Nfe_pad - Nfe_k) ÷ 2 + 1
        @views ksp_padded[pe0:pe0+Npe_k-1, fe0:fe0+Nfe_k-1] = ksp_w
        ksp_w = ksp_padded
    end
    img = fftshift(ifft(ifftshift(ksp_w, (1, 2)), (1, 2)), (1, 2))
    phase_sensitive ? Float32.(real.(img)) : Float32.(abs.(img))
end
