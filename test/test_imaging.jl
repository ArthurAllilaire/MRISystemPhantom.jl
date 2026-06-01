@testset "imaging pipeline" begin

    @testset "add_noise! — complex k-space" begin
        ksp = zeros(ComplexF32, 8, 16)
        rng = MersenneTwister(42)
        add_noise!(ksp, 1.0f0; rng = rng)
        # Noise was added
        @test any(ksp .!= 0)
        # Both channels carry noise
        @test any(real.(ksp) .!= 0)
        @test any(imag.(ksp) .!= 0)
        # No-op when σ ≤ 0
        ksp2 = zeros(ComplexF32, 4, 4)
        add_noise!(ksp2, 0.0f0)
        @test all(ksp2 .== 0)
        add_noise!(ksp2, -1.0f0)
        @test all(ksp2 .== 0)
        # Returns the array
        ksp3 = zeros(ComplexF32, 2, 2)
        @test add_noise!(ksp3, 0.5f0) === ksp3
    end

    @testset "add_gaussian_noise! — real array" begin
        x = zeros(Float32, 16)
        add_gaussian_noise!(x, 1.0f0; rng = MersenneTwister(1))
        @test any(x .!= 0)
        # No-op when σ ≤ 0
        x2 = zeros(Float32, 4)
        add_gaussian_noise!(x2, 0.0f0)
        @test all(x2 .== 0)
    end

    @testset "raw_to_kspace — shape and values" begin
        Npe, Nfe = 4, 8
        ksp = zeros(ComplexF32, Npe, Nfe)
        rng = MersenneTwister(7)
        add_noise!(ksp, 1.0f0; rng = rng)

        # Build a minimal mock raw object that mimics KomaMRI's output.
        struct _Profile data::Matrix{ComplexF64} end
        struct _RawAcq profiles::Vector{_Profile} end

        profiles = [_Profile(Matrix{ComplexF64}(reshape(ComplexF64.(ksp[k, :]), Nfe, 1)))
                    for k in 1:Npe]
        raw = _RawAcq(profiles)

        out = raw_to_kspace(raw, Npe, Nfe)
        @test size(out) == (Npe, Nfe)
        @test eltype(out) == ComplexF32
        for k in 1:Npe
            @test isapprox(out[k, :], ComplexF32.(ksp[k, :]); atol = 1e-6)
        end

        # Short raw (fewer profiles than Npe) fills missing rows with zero
        raw_short = _RawAcq(profiles[1:2])
        out_short = raw_to_kspace(raw_short, Npe, Nfe)
        @test all(out_short[3:end, :] .== 0)
    end

    @testset "kspace_to_image — magnitude and phase-sensitive" begin
        Npe, Nfe = 16, 32
        # DC impulse → flat image (all pixels equal constant)
        ksp = zeros(ComplexF32, Npe, Nfe)
        ksp[1, 1] = ComplexF32(1.0)  # unshifted DC
        img = kspace_to_image(ksp; phase_sensitive = false)
        @test size(img) == (Npe, Nfe)
        @test eltype(img) == Float32
        @test all(img .>= 0)
        @test isapprox(maximum(img), minimum(img); rtol = 1e-5)

        # Phase-sensitive returns real image (may be negative)
        img_ps = kspace_to_image(ksp; phase_sensitive = true)
        @test eltype(img_ps) == Float32

        # Round-trip: ifft(fft(x)) ≈ x (magnitude, up to pixel permutation)
        ksp2 = randn(MersenneTwister(3), ComplexF32, Npe, Nfe)
        img2 = kspace_to_image(ksp2; phase_sensitive = false)
        @test all(img2 .>= 0)
        # Energy is conserved (Parseval, up to IFFT normalisation)
        @test isapprox(sum(abs2.(ksp2)) / (Npe * Nfe),
                       sum(abs2.(img2));
                       rtol = 1e-4)
    end

    @testset "k-space phase-encode steps — even convention" begin
        FOV = 0.2

        for Npe in [8, 16, 32, 64]
            Δky = 1.0 / FOV
            ky_steps = [(k - 1 - Npe ÷ 2) * Δky for k in 1:Npe]

            # Correct number of steps
            @test length(ky_steps) == Npe

            # Range is exactly -Npe/2 to Npe/2 - 1 (in units of Δky)
            @test isapprox(ky_steps[1],   (-Npe ÷ 2) * Δky;      atol = 1e-12)
            @test isapprox(ky_steps[end], (Npe ÷ 2 - 1) * Δky;   atol = 1e-12)

            # Exactly one DC row (ky = 0), at index Npe÷2+1
            dc_idx = findfirst(s -> isapprox(s, 0.0; atol = 1e-12), ky_steps)
            @test dc_idx !== nothing
            @test dc_idx == Npe ÷ 2 + 1
            @test count(s -> isapprox(s, 0.0; atol = 1e-12), ky_steps) == 1

            # Uniform spacing
            diffs = diff(ky_steps)
            @test all(isapprox.(diffs, Δky; atol = 1e-12))
        end
    end

end
