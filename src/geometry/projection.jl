"""
    phys_to_pixel(x, N, FOV) → Int

Convert a physical coordinate `x` [m] to a 1-based Julia pixel index on a
centred grid of size `N` with field-of-view `FOV` [m]. The image centre
(x = 0) maps to pixel `N÷2 + 1`, matching the convention of `kspace_to_image`
(DC at index `N÷2+1` after fftshift).

Throws an error if `x` is outside the FOV (i.e. the pixel would be out of
bounds). Use `phys_to_pixel_wrap` if you intentionally want aliasing.
"""
function phys_to_pixel(x::Real, N::Int, FOV::Real)::Int
    px = round(Int, x * N / FOV) + N ÷ 2 + 1
    1 ≤ px ≤ N || error("Physical coordinate $x m is outside FOV=$FOV m (grid size $N)")
    px
end

"""
    phys_to_pixel_wrap(x, N, FOV) → Int

Like `phys_to_pixel` but wraps out-of-bounds coordinates modulo `N` instead
of erroring. Use this when modelling MRI FOV aliasing (e.g. `phantom_occupancy`
where spins outside the FOV fold back into the image).
"""
function phys_to_pixel_wrap(x::Real, N::Int, FOV::Real)::Int
    mod(round(Int, x * N / FOV) + N ÷ 2, N) + 1
end

"""
    phantom_parameter_map(phantom, values, Npe, Nfe, FOV) → Matrix{Float64}

Project per-spin scalar `values` onto a 2-D image grid (Npe × Nfe) and return
the mean value per pixel. Pixels with no spins are left at 0.0.

Typical use:
  `phantom_parameter_map(phantom, phantom.T1, Npe, Nfe, FOV)` → true T1 map
  `phantom_parameter_map(phantom, phantom.ρ,  Npe, Nfe, FOV)` → PD map

Uses the same centred-indexing and wrap-on-alias convention as
`phantom_occupancy`.
"""
function phantom_parameter_map(phantom, values::AbstractVector{<:Real},
                                Npe::Int, Nfe::Int, FOV::Real)::Matrix{Float64}
    pmap  = zeros(Float64, Npe, Nfe)
    count = zeros(Int,     Npe, Nfe)
    for k in eachindex(phantom.x)
        ife = phys_to_pixel_wrap(phantom.x[k], Nfe, FOV)
        ipe = phys_to_pixel_wrap(phantom.y[k], Npe, FOV)
        pmap[ipe, ife]  += values[k]
        count[ipe, ife] += 1
    end
    for idx in eachindex(pmap)
        count[idx] > 0 && (pmap[idx] /= count[idx])
    end
    pmap
end

"""
    image_to_kspace(img) → Matrix{ComplexF64}

Compute theoretical 2-D k-space from an image or occupancy grid via a centred
FFT: `fftshift(fft(ifftshift(img)))`. This is the forward transform that
matches the inverse used by `kspace_to_image` (ifftshift → ifft → fftshift).
"""
function image_to_kspace(img::AbstractMatrix)::Matrix{ComplexF64}
    fftshift(fft(ifftshift(img)))
end

"""
    phantom_occupancy(phantom, Npe, Nfe, FOV) → Matrix{Float64}

Project phantom spin positions onto a 2-D image pixel grid and count spins per
pixel. Uses the same centred-indexing convention as `kspace_to_image`:
DC at array index (Npe÷2+1, Nfe÷2+1). Spins outside the FOV wrap modulo the
grid (models MRI aliasing).

`phantom.x` maps to the frequency-encode (column) axis; `phantom.y` maps to
the phase-encode (row) axis.
"""
function phantom_occupancy(phantom, Npe::Int, Nfe::Int, FOV::Real)
    occ = zeros(Float64, Npe, Nfe)
    for k in eachindex(phantom.x)
        ife = phys_to_pixel_wrap(phantom.x[k], Nfe, FOV)
        ipe = phys_to_pixel_wrap(phantom.y[k], Npe, FOV)
        occ[ipe, ife] += 1.0
    end
    occ
end

"""
    phantom_occupancy(phantom, Npe, Nfe, Nsl, FOV_xy, FOV_z) → Array{Float64,3}

3-D overload. Projects phantom spins onto a (Npe × Nfe × Nsl) voxel grid.
`phantom.x` → frequency-encode, `phantom.y` → phase-encode,
`phantom.z` → slice axis. Spins outside the FOV wrap modulo their respective
grid (models MRI aliasing).
"""
function phantom_occupancy(phantom, Npe::Int, Nfe::Int, Nsl::Int,
                            FOV_xy::Real, FOV_z::Real)
    occ = zeros(Float64, Npe, Nfe, Nsl)
    for k in eachindex(phantom.x)
        ife = phys_to_pixel_wrap(phantom.x[k], Nfe, FOV_xy)
        ipe = phys_to_pixel_wrap(phantom.y[k], Npe, FOV_xy)
        isl = phys_to_pixel_wrap(phantom.z[k], Nsl, FOV_z)
        occ[ipe, ife, isl] += 1.0
    end
    occ
end
