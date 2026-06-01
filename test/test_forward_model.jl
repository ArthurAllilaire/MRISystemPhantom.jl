using Test
using Random
using FFTW: fft, ifft, fftshift, ifftshift
import Statistics: mean, std

# Forward models: the band-limiting operator (`bandlimit_image`) and the two
# IR-SE forward models. Core claims under test:
#   1. central_crop is symmetric about DC and bounds-checked.
#   2. bandlimit_image conserves total intensity (Σ image == Σ fine), returns a
#      real image for real input, and is a no-op when crop size == input size.
#   3. bandlimit_image removes high-frequency (out-of-band) content — the moiré
#      mechanism — while preserving a smooth (in-band) field.
#   4. ir_se_theory_image gives a far smoother uniform region than the naive
#      ir_se_theory_image_binned (the whole point of the band-limited model),
#      while conserving total signal.

@testset "forward model" begin
    @testset "central_crop centring + bounds" begin
        A = reshape(collect(1:64), 8, 8) .|> Float64
        c = MRISystemPhantom.central_crop(A, 4, 4)
        @test size(c) == (4, 4)
        # DC of an 8-grid is index 5; a symmetric 4-crop keeps 3:6 (DC lands at
        # the crop's own DC index 4÷2+1 = 3).
        @test c == A[3:6, 3:6]
        @test_throws ErrorException MRISystemPhantom.central_crop(A, 10, 4)
    end

    @testset "bandlimit_image: identity + intensity + realness" begin
        rng = MersenneTwister(1)
        img = rand(rng, 16, 16)
        # Crop size == input size ⇒ round-trip identity (fft then ifft). ksp is
        # stored ComplexF32 (matching the imaging pipeline), so tolerate Float32.
        bl0 = bandlimit_image(img, 16, 16)
        @test bl0.image ≈ img atol = 1e-4
        # Downsample by cropping k-space: total intensity conserved (Σ = K_DC).
        bl = bandlimit_image(img, 8, 8)
        @test size(bl.image) == (8, 8)
        @test sum(bl.image) ≈ sum(img) rtol = 1e-4
        # Real input ⇒ real recon (symmetric crop preserves conjugate symmetry).
        @test bl.image isa Matrix{Float64}
        @test all(isfinite, bl.image)
    end

    @testset "bandlimit_image kills out-of-band, keeps in-band" begin
        N, Nhi = 16, 64
        xs = range(0, 2π; length = Nhi + 1)[1:Nhi]
        # Low-frequency field (1 cycle across FOV) — well inside the N=16 band.
        low  = [cos(x) for x in xs, _ in xs]
        # High-frequency field (Nhi/2 cycles) — far above kmax for N=16.
        high = [cos((Nhi ÷ 2) * x) for x in xs, _ in xs]
        bl_low  = bandlimit_image(low,  N, N)
        bl_high = bandlimit_image(high, N, N)
        # In-band content survives with real amplitude; out-of-band is removed.
        @test maximum(abs, bl_low.image)  > 0.1 * maximum(abs, low)
        @test maximum(abs, bl_high.image) < 1e-6 * maximum(abs, high)
    end

    @testset "ir_se_theory_image smoother than binned, same total" begin
        cfg = PhantomConfig(field = :T15, voxel_size_mm = 1.0,
                            include_plates = [:water],
                            slice_thickness_mm = 1.0,
                            slice_center_mm = (0.0, 0.0, MRISystemPhantom.PLATE_Z_MM.T1))
        phantom = build_phantom(cfg)
        FOV, Npe, Nfe = 0.2, 32, 64
        kw = (TI = 0.1, TR = 5.0, α_exc = π/2, θ_inv = π, FOV = FOV,
              Npe = Npe, Nfe = Nfe)

        binned = ir_se_theory_image_binned(phantom; kw...)
        band   = ir_se_theory_image(phantom; kw..., voxel_mm = 1.0)

        @test size(band.image) == (Npe, Nfe)
        # Both forward models conserve the same total signal (each spin once).
        @test sum(band.image) ≈ sum(binned.image) rtol = 1e-6

        # Smoothness on a uniform water patch (no spheres): the band-limited
        # model must have a much lower coefficient of variation than the binned
        # one, whose lattice moiré inflates it.
        patch = (6:11, 25:45)
        cov(a) = std(vec(a)) / abs(mean(vec(a)))
        @test cov(abs.(band.image[patch...])) < 0.5 * cov(abs.(binned.image[patch...]))

        # Regression: at a larger recon grid (wider kmax) an *incommensurate*
        # fine grid would let the grid-beat back into the band and inflate the
        # CoV above the binned model. The commensurate fine grid must stay smooth.
        Npe2, Nfe2 = 64, 128
        kw2 = (TI = 0.1, TR = 5.0, α_exc = π/2, θ_inv = π, FOV = FOV,
               Npe = Npe2, Nfe = Nfe2)
        binned2 = ir_se_theory_image_binned(phantom; kw2...)
        band2   = ir_se_theory_image(phantom; kw2..., voxel_mm = 1.0)
        patch2  = (12:18, 50:90)
        @test cov(abs.(band2.image[patch2...])) < 0.7 * cov(abs.(binned2.image[patch2...]))
    end
end
