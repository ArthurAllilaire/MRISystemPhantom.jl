using Statistics: mean, std
using LinearAlgebra: I, det

@testset "augmentations" begin
    cfg_nojitter = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:T1])

    @testset "rotation matrix is orthogonal" begin
        R = rotation_matrix(0.3, -0.7, 1.1)
        @test isapprox(R * R', Matrix(1.0I, 3, 3); atol = 1e-10)
        @test isapprox(det(R), 1.0; atol = 1e-10)
        @test isapprox(rotation_matrix(0, 0, 0), Matrix(1.0I, 3, 3); atol = 1e-12)
    end

    @testset "rotation leaves |r| invariant" begin
        obj = build_phantom(cfg_nojitter)
        r_before = sqrt.(obj.x.^2 .+ obj.y.^2 .+ obj.z.^2)
        cfg_r = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:T1],
                              rotation = (0.0, 0.0, π/4))
        obj_r = build_phantom(cfg_r)
        r_after = sqrt.(obj_r.x.^2 .+ obj_r.y.^2 .+ obj_r.z.^2)
        @test length(r_before) == length(r_after)
        @test isapprox(sort(r_before), sort(r_after); atol = 1e-9)
    end

    @testset "translation shifts all spins" begin
        cfg_t = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:T1],
                              translation_mm = (10.0, -5.0, 2.0))
        obj   = build_phantom(cfg_nojitter)
        obj_t = build_phantom(cfg_t)
        @test isapprox(mean(obj_t.x) - mean(obj.x),  10e-3;  atol = 1e-10)
        @test isapprox(mean(obj_t.y) - mean(obj.y), -5e-3;   atol = 1e-10)
        @test isapprox(mean(obj_t.z) - mean(obj.z),  2e-3;   atol = 1e-10)
    end

    @testset "T1/T2 jitter expected stddev" begin
        cfg_j = PhantomConfig(
            voxel_size_mm = 2.0, include_plates = [:T1],
            augment = AugmentConfig(T1_sigma_rel = 0.10, T2_sigma_rel = 0.10),
            rng_seed = 123)
        obj = build_phantom(cfg_j)

        # Spins whose *mean* T1 still lives in the top sphere's band should
        # show ~10 % relative scatter.
        t1_max = maximum(T1_ARRAY[:T3])
        in_sphere = abs.(obj.T1 .- t1_max) .<= 0.30 * t1_max
        @test any(in_sphere)
        σ_empirical_rel = std(obj.T1[in_sphere]) / mean(obj.T1[in_sphere])
        @test 0.04 < σ_empirical_rel < 0.18
    end

    @testset "PD jitter stays clipped to [0,1]" begin
        cfg_j = PhantomConfig(
            voxel_size_mm = 3.0, include_plates = [:PD],
            augment = AugmentConfig(PD_sigma_abs = 2.0),   # huge σ
            rng_seed = 7)
        obj = build_phantom(cfg_j)
        @test all(0.0 .<= obj.ρ .<= 1.0)
    end

    @testset "B0 jitter sets delta_w" begin
        cfg_j = PhantomConfig(
            voxel_size_mm = 3.0, include_plates = [:T1],
            augment = AugmentConfig(B0_sigma_Hz = 20.0),
            rng_seed = 99)
        obj = build_phantom(cfg_j)
        @test std(obj.Δw) > 0
    end
end
