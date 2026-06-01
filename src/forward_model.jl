# Analytical forward models: phantom → predicted IR-SE image.
#
# Two models live here. `ir_se_theory_image` (band-limited) is the one to use;
# `ir_se_theory_image_binned` (naive per-pixel binning) is kept only as the
# reference that motivates it — see the docstrings for the moiré story.

"""
    central_crop(A, np, nf) -> SubMatrix

Crop the central `np × nf` block of a centred (fftshifted) spectrum `A`. DC sits
at index `N÷2+1`, so a symmetric crop preserves the conjugate symmetry of a real
image's spectrum (⇒ a real image after IFFT).
"""
function central_crop(A::AbstractMatrix, np::Int, nf::Int)
    Np, Nf = size(A)
    (np ≤ Np && nf ≤ Nf) || error("crop size ($np×$nf) exceeds spectrum ($Np×$Nf)")
    p0 = (Np ÷ 2 + 1) - (np ÷ 2)
    f0 = (Nf ÷ 2 + 1) - (nf ÷ 2)
    A[p0:p0+np-1, f0:f0+nf-1]
end

"""
    bandlimit_image(img_fine, Npe, Nfe) -> (image, ksp)

Band-limit a fine-grid image to an `Npe × Nfe` recon grid by
`FFT → central crop → IFFT`. This reproduces the image-formation operator of a
real acquisition: a finite k-space window convolves the object with a
Dirichlet/sinc PSF (smooth interiors, Gibbs ringing at edges). Out-of-band
content is *discarded*, not aliased.

`image_to_kspace` uses an unnormalised `fft` and `kspace_to_image` carries the
`1/N` of `ifft` over the *coarse* (crop) grid, so total intensity is conserved
(`Σ image = K_DC = Σ img_fine`); amplitudes need no manual rescale. A real
`img_fine` ⇒ conjugate-symmetric spectrum ⇒ a symmetric crop reconstructs a
clean real image. `ksp` is the cropped (band-limited) k-space; `image` is the
signed real recon.
"""
function bandlimit_image(img_fine::AbstractMatrix{<:Real}, Npe::Int, Nfe::Int)
    K   = image_to_kspace(img_fine)
    ksp = Matrix{ComplexF32}(central_crop(K, Npe, Nfe))
    img = Float64.(kspace_to_image(ksp; phase_sensitive = true))
    (image = img, ksp = ksp)
end

"""
    ir_se_theory_image_binned(phantom; TI, TR, α_exc, θ_inv, FOV, Npe, Nfe)
        -> (image, ksp, T1_map, rho_map)

NAIVE per-pixel forward model, kept for reference only — **superseded by**
`ir_se_theory_image`. Bins phantom spins into the `Npe × Nfe` recon grid and
sets each pixel to `transient_mz_at_excite_npe(T1̄, …) · ρ̄ · sin(α_exc) ·
(spin count)`.

Why it's wrong: the phantom is a regular voxel lattice whose spacing is
incommensurate with the recon pixel pitch (`FOV/N`), so the integer spin count
per pixel beats periodically → strong horizontal/vertical **moiré** streaks. The
lattice frequency (~1000 cycles/m for 1 mm voxels) sits far above the imaging
band-limit `kmax = N/(2·FOV)`; binning *aliases* that out-of-band energy back
into the image as the streaks. `ir_se_theory_image` fixes this by band-limiting
instead of binning.
"""
function ir_se_theory_image_binned(phantom; TI, TR, α_exc, θ_inv, FOV, Npe, Nfe)
    T1_map  = phantom_parameter_map(phantom, phantom.T1, Npe, Nfe, FOV)
    rho_map = phantom_parameter_map(phantom, phantom.ρ,  Npe, Nfe, FOV)
    occ_map = phantom_occupancy(phantom, Npe, Nfe, FOV)
    img = zeros(Float64, Npe, Nfe)
    for ipe in 1:Npe, ife in 1:Nfe
        T1_px = T1_map[ipe, ife]
        T1_px > 0 || continue
        img[ipe, ife] = transient_mz_at_excite_npe(
            T1_px, TI, TR, θ_inv, α_exc; Npe = Npe
        ) * rho_map[ipe, ife] * sin(α_exc) * occ_map[ipe, ife]
    end
    ksp = Matrix{ComplexF32}(image_to_kspace(img))
    (image = img, ksp = ksp, T1_map = T1_map, rho_map = rho_map)
end

"""
    ir_se_theory_image(phantom; TI, TR, α_exc, θ_inv, FOV, Npe, Nfe, voxel_mm)
        -> (image, ksp, T1_map, rho_map)

Band-limited analytical forward model for one IR-SE acquisition, built to share
the **simulator's** image-formation operator (finite-k-space truncation) rather
than naive pixel binning — so it carries the same Gibbs/sinc PSF as the KomaMRI
recon and has no lattice moiré (cf. `ir_se_theory_image_binned`).

Steps:
1. Build the signal map `transient_mz_at_excite_npe(T1̄, …)·ρ̄·sin(α_exc)·count`
   on a FINE grid sized as an integer multiple of the voxel count across the FOV
   (`m·n_vox`, so the grid is *commensurate* with the voxel lattice — see the
   body for why an incommensurate fine grid reintroduces a beat/streak).
2. `bandlimit_image` it: FFT → crop the central `Npe×Nfe` k-space → IFFT.

The crop is the band-limiting (keep `|k| ≤ kmax`, discard the rest) — the only
step the naive model gets wrong. Total intensity is conserved, so amplitudes
match the sim with no rescale.

Any remaining theory−sim difference is then genuine physics the per-pixel model
omits (per-shot transient k-space modulation, spoiling, slice profile,
off-resonance) — which is what a sim−theory diff should isolate.

`T1_map`/`rho_map` are returned on the coarse `Npe×Nfe` grid (for parameter-map
display); `image`/`ksp` come from the fine→crop path. `voxel_mm` sizes the fine
grid (the phantom's voxel pitch).
"""
function ir_se_theory_image(phantom; TI, TR, α_exc, θ_inv, FOV, Npe, Nfe, voxel_mm)
    # Coarse parameter maps — for the T1/ρ display panels only.
    T1_map  = phantom_parameter_map(phantom, phantom.T1, Npe, Nfe, FOV)
    rho_map = phantom_parameter_map(phantom, phantom.ρ,  Npe, Nfe, FOV)

    # Fine grid must be COMMENSURATE with the voxel lattice: the voxel pitch has
    # to be an integer number of fine pixels (fine size = m·n_vox, where
    # n_vox = FOV/voxel is the voxel count across the FOV). Otherwise the two
    # grids beat at the low frequency |1/fine_pitch − 1/voxel|, which can fall
    # *inside* the recon band kmax = N/(2·FOV) and survive the crop — reappearing
    # as exactly the streaks we set out to remove (this bites harder at larger N,
    # whose wider kmax admits the beat). `m` is the smallest factor making the
    # fine grid at least as large as the recon grid; n_vox itself (m=1) already
    # gives fine_pitch = voxel, i.e. the phantom's own discretisation.
    fov_mm = FOV * 1000
    n_vox  = round(Int, fov_mm / voxel_mm)
    n_vox ≥ 1 || error("voxel_mm ($voxel_mm) larger than FOV ($fov_mm mm)")
    m      = max(1, cld(max(Npe, Nfe), n_vox))
    Nfe_hi = m * n_vox
    Npe_hi = m * n_vox

    T1_hi  = phantom_parameter_map(phantom, phantom.T1, Npe_hi, Nfe_hi, FOV)
    rho_hi = phantom_parameter_map(phantom, phantom.ρ,  Npe_hi, Nfe_hi, FOV)
    occ_hi = phantom_occupancy(phantom, Npe_hi, Nfe_hi, FOV)
    theory_hi = zeros(Float64, Npe_hi, Nfe_hi)
    for ipe in 1:Npe_hi, ife in 1:Nfe_hi
        T1_px = T1_hi[ipe, ife]
        T1_px > 0 || continue
        theory_hi[ipe, ife] = transient_mz_at_excite_npe(
            T1_px, TI, TR, θ_inv, α_exc; Npe = Npe
        ) * rho_hi[ipe, ife] * sin(α_exc) * occ_hi[ipe, ife]
    end

    bl = bandlimit_image(theory_hi, Npe, Nfe)
    (image = bl.image, ksp = bl.ksp, T1_map = T1_map, rho_map = rho_map)
end
