@testset "determinism" begin
    cfg = PhantomConfig(
        voxel_size_mm = 3.0,
        include_plates = [:T1, :T2, :PD, :fiducials],
        augment = AugmentConfig(
            T1_sigma_rel      = 0.02,
            T2_sigma_rel      = 0.02,
            PD_sigma_abs      = 0.01,
            position_sigma_mm = 0.2,
            B0_sigma_Hz       = 3.0,
            drop_sphere_p     = 0.05,
        ),
        rng_seed = 42,
    )

    a = build_phantom(cfg)
    b = build_phantom(cfg)

    @test a.x == b.x
    @test a.y == b.y
    @test a.z == b.z
    @test a.T1 == b.T1
    @test a.T2 == b.T2
    @test a.ρ  == b.ρ
    @test a.Δw == b.Δw

    # Different seed → different draws
    c = build_phantom(PhantomConfig(
        voxel_size_mm = 3.0,
        include_plates = cfg.include_plates,
        augment       = cfg.augment,
        rng_seed      = cfg.rng_seed + 1,
    ))
    @test a.T1 != c.T1
end
