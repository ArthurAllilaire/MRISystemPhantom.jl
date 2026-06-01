_one_t1_descriptor() = sphere_descriptors(:T1, PhantomConfig())[1]

@testset "builder" begin
    cfg = PhantomConfig(voxel_size_mm = 3.0)   # coarse, fast

    @testset "sphere descriptors" begin
        t1s  = sphere_descriptors(:T1, cfg)
        t2s  = sphere_descriptors(:T2, cfg)
        pds  = sphere_descriptors(:PD, cfg)
        fids = sphere_descriptors(:fiducials, cfg)

        @test length(t1s)  == 14
        @test length(t2s)  == 14
        @test length(pds)  == 14
        @test length(fids) == 57
        @test length(t1s) + length(t2s) + length(pds) == 42

        # Labels are unique within each plate
        for descs in (t1s, t2s, pds, fids)
            @test length(unique(d -> d.label, descs)) == length(descs)
        end

        # Contrast sphere radii
        @test all(d -> d.radius == CONTRAST_RADIUS_M, t1s)
        @test all(d -> d.radius == CONTRAST_RADIUS_M, t2s)
        @test all(d -> d.radius == CONTRAST_RADIUS_M, pds)
        @test all(d -> d.radius == FIDUCIAL_RADIUS_M, fids)

        # T1 descriptor T1 values match manual (cfg defaults to :T15)
        @test [d.T1 for d in t1s] == T1_ARRAY[:T15]
        # Field :T3 picks the 3T table
        @test [d.T1 for d in sphere_descriptors(:T1, PhantomConfig(field = :T3))] == T1_ARRAY[:T3]

        # PD descriptor ρ values match H₂O fractions
        @test [d.ρ for d in pds] == PD_FRACTIONS

        @test_throws ErrorException sphere_descriptors(:bogus, cfg)
    end

    @testset "descriptor helpers" begin
        d = SphereDescriptor((0.010, 0.0, 0.020), CONTRAST_RADIUS_M, 0.8,
                             0.5, 0.05, 0.04, 2.0, :probe)
        d_relaxed = with_sphere_relaxation(d, 0.7, 0.07)
        @test d_relaxed.centre == d.centre
        @test d_relaxed.radius == d.radius
        @test d_relaxed.ρ == d.ρ
        @test d_relaxed.delta_w == d.delta_w
        @test d_relaxed.label == d.label
        @test d_relaxed.T1 == 0.7
        @test d_relaxed.T2 == 0.07
        @test d_relaxed.T2s == 0.07

        d_tx = transform_descriptor(d, (0.0, 0.0, π / 2), (0.001, 0.002, 0.003))
        @test isapprox(d_tx.centre[1], 0.001; atol = 1e-12)
        @test isapprox(d_tx.centre[2], 0.012; atol = 1e-12)
        @test isapprox(d_tx.centre[3], 0.023; atol = 1e-12)
        @test d_tx.radius == d.radius
        @test d_tx.T1 == d.T1

        @test sphere_descriptor_pixel(
            SphereDescriptor((0.0, 0.0, 0.0), CONTRAST_RADIUS_M, 1.0,
                             1.0, 0.1, 0.1, 0.0, :origin),
            8, 16, 0.2,
        ) == (5, 9)
    end

    @testset "build_sphere produces a populated Phantom" begin
        d = _one_t1_descriptor()
        p = build_sphere(d, 1e-3)
        @test length(p.x) > 0
        @test length(p.x) == length(p.T1) == length(p.T2) == length(p.ρ)
        @test all(p.T1 .== d.T1)
        @test all(p.T2 .== d.T2)
    end

    @testset "build_plate concatenates sphere voxels" begin
        p = build_plate(:T1, cfg)
        @test length(p.x) > 0
        # 14 T1 spheres should produce spins spanning a range of T1 values
        @test length(unique(p.T1)) == 14
        # z positions all on the T1 plate, within a sphere-radius margin
        @test all(abs.(p.z .- PLATE_Z_MM.T1 * 1e-3) .<= CONTRAST_RADIUS_M + 1e-9)
    end

    @testset "build_phantom full assembly" begin
        obj = build_phantom(cfg)
        @test length(obj.x) > 0
        @test length(obj.x) == length(obj.T1) == length(obj.T2)
        # With water included, a lot more spins than plates-only
        obj_noWater = build_phantom(PhantomConfig(
            voxel_size_mm = 3.0,
            include_plates = [:T1, :T2, :PD, :fiducials]))
        @test length(obj.x) > length(obj_noWater.x)
        # Named
        @test obj.name == "qalibremd"
    end

    @testset "plate subset" begin
        cfg2 = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:T1])
        obj = build_phantom(cfg2)
        # Only T1 array → unique T1 values is exactly 14
        @test length(unique(obj.T1)) == 14
    end

    @testset "field selection matters" begin
        cfgA = PhantomConfig(voxel_size_mm = 3.0, field = :T3,
                             include_plates = [:T1])
        cfgB = PhantomConfig(voxel_size_mm = 3.0, field = :T15,
                             include_plates = [:T1])
        A = build_phantom(cfgA)
        B = build_phantom(cfgB)
        @test sort(unique(A.T1)) != sort(unique(B.T1))
    end

    @testset "drop_sphere_p removes spheres" begin
        cfgD = PhantomConfig(
            voxel_size_mm = 3.0,
            include_plates = [:T1],
            augment = AugmentConfig(drop_sphere_p = 1.0))   # drop all
        descs = sphere_descriptors(:T1, cfgD)
        @test isempty(descs)
    end

    @testset "sphere label filters and explicit descriptors" begin
        cfg_keep = PhantomConfig(include_plates = [:T1],
                                 keep_sphere_labels = [:T1_1, :T1_3])
        @test [d.label for d in sphere_descriptors(:T1, cfg_keep)] == [:T1_1, :T1_3]

        cfg_drop = PhantomConfig(include_plates = [:T1],
                                 drop_sphere_labels = [:T1_1, :T1_3])
        labels_drop = [d.label for d in sphere_descriptors(:T1, cfg_drop)]
        @test length(labels_drop) == 12
        @test :T1_1 ∉ labels_drop
        @test :T1_3 ∉ labels_drop

        custom = SphereDescriptor((0.0, 0.0, 0.0), CONTRAST_RADIUS_M, 1.0,
                                  0.42, 0.084, 0.084, 0.0, :custom)
        cfg_custom = PhantomConfig(voxel_size_mm = 3.0,
                                   include_plates = Symbol[],
                                   custom_sphere_descriptors = [custom])
        obj_custom = build_phantom(cfg_custom)
        @test length(obj_custom.x) > 0
        @test unique(obj_custom.T1) == [0.42]
        @test [d.label for d in all_sphere_descriptors(cfg_custom)] == [:custom]

        cfg_custom_water = PhantomConfig(voxel_size_mm = 3.0,
                                         include_plates = [:water],
                                         custom_sphere_descriptors = [custom],
                                         slice_thickness_mm = 3.0,
                                         slice_center_mm = (0.0, 0.0, 0.0))
        obj_custom_water = build_phantom(cfg_custom_water)
        @test length(obj_custom_water.x) > length(obj_custom.x)
        @test any(obj_custom_water.T1 .== 0.42)
    end

    @testset "water fills omitted sphere volumes" begin
        base = sphere_descriptors(
            :T1, PhantomConfig(field = :T15, include_plates = [:T1]))
        kept = base[1]
        omitted = base[2]
        cfg_keep_water = PhantomConfig(
            voxel_size_mm = 2.0,
            include_plates = [:T1, :water],
            keep_sphere_labels = [kept.label],
            slice_thickness_mm = 2.0,
            slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        water = build_background_water(cfg_keep_water)

        inside_kept = @. ((water.x - kept.centre[1])^2 +
                          (water.y - kept.centre[2])^2 +
                          (water.z - kept.centre[3])^2) <= kept.radius^2 - 1e-12
        @test !any(inside_kept)

        inside_omitted = @. ((water.x - omitted.centre[1])^2 +
                             (water.y - omitted.centre[2])^2 +
                             (water.z - omitted.centre[3])^2) <= omitted.radius^2
        @test any(inside_omitted)
    end

    @testset "slice_thickness_mm — phantom-attached slab mask" begin
        # Slab centred on T1 plate keeps only T1-plate spins.
        cfgS = PhantomConfig(voxel_size_mm = 0.5,
                             include_plates = [:T1, :T2, :PD],
                             slice_thickness_mm = 1.0,
                             slice_center_mm    = (0.0, 0.0, PLATE_Z_MM.T1))
        obj = build_phantom(cfgS)
        @test length(obj.x) > 0
        @test all(abs.(obj.z .- PLATE_Z_MM.T1 * 1e-3) .<= 0.5e-3 + 1e-12)
        # Every surviving T1 value must be one of the T1-plate values.
        t1_vals = sort(unique([d.T1 for d in sphere_descriptors(:T1, cfgS)]))
        @test all(in(t1_vals), obj.T1)

        # Same slab but excluding the T1 plate → no spins survive.
        cfgEmpty = PhantomConfig(voxel_size_mm = 0.5,
                                 include_plates = [:T2, :PD],
                                 slice_thickness_mm = 1.0,
                                 slice_center_mm    = (0.0, 0.0, PLATE_Z_MM.T1))
        @test length(build_phantom(cfgEmpty).x) == 0

        # Default (nothing) reproduces the full phantom z-extent.
        full = build_phantom(PhantomConfig(voxel_size_mm = 3.0))
        @test maximum(abs.(full.z)) > 1e-2   # ≫ 1 mm

        # Phantom-attached semantics: rotating the phantom rotates the slab.
        cfgRot = PhantomConfig(voxel_size_mm = 1.0,
                               include_plates = [:water],
                               slice_thickness_mm = 1.0,
                               slice_center_mm    = (0.0, 0.0, 0.0),
                               rotation           = (deg2rad(20.0), 0.0, 0.0))
        objRot = build_phantom(cfgRot)
        @test length(objRot.x) > 0
        R = rotation_matrix(cfgRot.rotation...)
        n_scanner = R * [0.0, 0.0, 1.0]
        dists = @. objRot.x * n_scanner[1] + objRot.y * n_scanner[2] +
                   objRot.z * n_scanner[3]
        @test all(abs.(dists) .<= 0.5e-3 + 1e-9)
        @test maximum(abs.(objRot.z)) > 0.5e-3
    end

    @testset "slice mask — float tolerance at slab boundary" begin
        # Regression: when slice_thickness_mm == voxel_size_mm, voxel layers
        # land exactly on the slab edge. A strict ≤ comparison drops them via
        # IEEE-754 roundoff and the slab comes out empty. The builder uses a
        # 1 nm float tolerance to keep these boundary voxels.
        #
        # T1-plate spheres sit at z = 56.5 mm with radius 7.5 mm, so a 1 mm
        # voxel grid centred on the sphere has z-layers at …, 55, 56, 57, 58 …
        # The slab [56.0, 57.0] mm contains them only via the tolerance.
        cfg = PhantomConfig(voxel_size_mm = 1.0,
                            include_plates = [:T1, :water],
                            slice_thickness_mm = 1.0,
                            slice_center_mm    = (0.0, 0.0, PLATE_Z_MM.T1))
        obj = build_phantom(cfg)
        @test length(obj.x) > 0
        # Boundary layers (56 and 57 mm) should be present, not silently dropped.
        z_mm = obj.z .* 1000
        @test any(abs.(z_mm .- 56.0) .< 1e-6)
        @test any(abs.(z_mm .- 57.0) .< 1e-6)
    end

    @testset "background water excludes contrast volumes" begin
        cfgW = PhantomConfig(voxel_size_mm = 3.0,
                             include_plates = [:T1, :T2, :PD, :fiducials, :water])
        water = build_background_water(cfgW)
        # No water voxel should lie strictly inside any contrast/fiducial sphere
        for d in all_sphere_descriptors(cfgW)
            dx = water.x .- d.centre[1]
            dy = water.y .- d.centre[2]
            dz = water.z .- d.centre[3]
            @test !any(@.(dx^2 + dy^2 + dz^2) .<= d.radius^2 - 1e-12)
        end
    end

    @testset "coarse sliced water uses weighted plane sampling" begin
        cfg_fine = PhantomConfig(voxel_size_mm = 1.0,
                                 include_plates = [:water],
                                 water_voxel_size_mm = 1.0,
                                 slice_thickness_mm = 1.0,
                                 slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        cfg_coarse = PhantomConfig(voxel_size_mm = 1.0,
                                   include_plates = [:water],
                                   water_voxel_size_mm = 3.0,
                                   slice_thickness_mm = 1.0,
                                   slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        fine = build_background_water(cfg_fine)
        coarse = build_background_water(cfg_coarse)
        @test isapprox(length(coarse.x) / length(fine.x), 1 / 9; rtol = 0.25)
        @test isapprox(sum(coarse.ρ), sum(fine.ρ); rtol = 0.10)
        @test all(coarse.ρ .== BACKGROUND_WATER[cfg_coarse.field].ρ * 9.0)
    end

    @testset "sliced water can use stacked through-plane sheets" begin
        cfg_single = PhantomConfig(voxel_size_mm = 1.0,
                                   include_plates = [:water],
                                   water_voxel_size_mm = 3.0,
                                   slice_thickness_mm = 3.0,
                                   slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        cfg_stack = PhantomConfig(voxel_size_mm = 1.0,
                                  include_plates = [:water],
                                  water_voxel_size_mm = 3.0,
                                  water_throughplane_voxel_size_mm = 1.0,
                                  slice_thickness_mm = 3.0,
                                  slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        single = build_background_water(cfg_single)
        stack = build_background_water(cfg_stack)
        @test length(stack.x) > length(single.x)
        @test isapprox(sum(stack.ρ), sum(single.ρ); rtol = 0.10)
        @test sort(unique(round.(stack.z .* 1000; digits = 6))) == [55.5, 56.5, 57.5]
        @test all(stack.ρ .== BACKGROUND_WATER[cfg_stack.field].ρ * 9.0)
    end

    @testset "sliced water stack edge sheets are thickness weighted" begin
        cfg = PhantomConfig(voxel_size_mm = 1.0,
                            include_plates = [:water],
                            water_voxel_size_mm = 3.0,
                            water_throughplane_voxel_size_mm = 2.0,
                            slice_thickness_mm = 3.0,
                            slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        water = build_background_water(cfg)
        z_mm = round.(water.z .* 1000; digits = 6)
        @test sort(unique(z_mm)) == [56.0, 57.5]
        @test Set(unique(water.ρ)) == Set(BACKGROUND_WATER[cfg.field].ρ .* [9.0, 18.0])
    end

    @testset "coarse water preserves sphere prefix" begin
        dry_cfg = PhantomConfig(voxel_size_mm = 1.0,
                                include_plates = [:T1],
                                slice_thickness_mm = 1.0,
                                slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        full_cfg = PhantomConfig(voxel_size_mm = 1.0,
                                 include_plates = [:T1, :water],
                                 water_voxel_size_mm = 3.0,
                                 slice_thickness_mm = 1.0,
                                 slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        dry = build_phantom(dry_cfg)
        full = build_phantom(full_cfg)
        n_dry = length(dry.x)
        @test n_dry > 0
        @test length(full.x) > n_dry
        @test full.x[1:n_dry] ≈ dry.x
        @test full.y[1:n_dry] ≈ dry.y
        @test full.z[1:n_dry] ≈ dry.z
        @test full.ρ[1:n_dry] == dry.ρ
    end

    @testset "coarse water survives oblique pose on transformed plane" begin
        cfg = PhantomConfig(voxel_size_mm = 1.0,
                            include_plates = [:water],
                            water_voxel_size_mm = 3.0,
                            slice_thickness_mm = 1.0,
                            slice_normal = (0.0, 1.0, 1.0),
                            slice_center_mm = (0.0, 20.0 / sqrt(2), 20.0 / sqrt(2)),
                            rotation = (deg2rad(20.0), 0.0, deg2rad(10.0)),
                            translation_mm = (3.0, -2.0, 5.0))
        obj = build_phantom(cfg)
        @test length(obj.x) > 0
        n_p, _, _ = slice_basis(cfg.slice_normal)
        p_p = cfg.slice_center_mm .* 1e-3
        R = rotation_matrix(cfg.rotation...)
        t = collect(cfg.translation_mm .* 1e-3)
        p_s = R * collect(p_p) + t
        n_s = R * collect(n_p)
        dists = @. (obj.x - p_s[1]) * n_s[1] +
                   (obj.y - p_s[2]) * n_s[2] +
                   (obj.z - p_s[3]) * n_s[3]
        @test all(abs.(dists) .< 1e-9)
    end

    @testset "non-sliced coarse water is 3D weighted" begin
        cfg_fine = PhantomConfig(voxel_size_mm = 4.0,
                                 include_plates = [:water],
                                 water_voxel_size_mm = 4.0)
        cfg_coarse = PhantomConfig(voxel_size_mm = 4.0,
                                   include_plates = [:water],
                                   water_voxel_size_mm = 8.0)
        fine = build_background_water(cfg_fine)
        coarse = build_background_water(cfg_coarse)
        @test isapprox(length(coarse.x) / length(fine.x), 1 / 8; rtol = 0.25)
        @test isapprox(sum(coarse.ρ), sum(fine.ρ); rtol = 0.15)
        @test all(coarse.ρ .== BACKGROUND_WATER[cfg_coarse.field].ρ * 8.0)
    end

    @testset "coarse water validation and finer water" begin
        cfg_finer = PhantomConfig(voxel_size_mm = 3.0,
                                  include_plates = [:water],
                                  water_voxel_size_mm = 1.5,
                                  slice_thickness_mm = 3.0,
                                  slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1))
        finer = build_background_water(cfg_finer)
        @test length(finer.x) > 0
        @test all(finer.ρ .== BACKGROUND_WATER[cfg_finer.field].ρ * 0.25)

        cfg_bad_water = PhantomConfig(voxel_size_mm = 1.0,
                                      include_plates = [:water],
                                      water_voxel_size_mm = 0.0)
        @test_throws ErrorException build_background_water(cfg_bad_water)

        cfg_bad_through = PhantomConfig(voxel_size_mm = 1.0,
                                        include_plates = [:water],
                                        slice_thickness_mm = 1.0,
                                        water_throughplane_voxel_size_mm = 0.0)
        @test_throws ErrorException build_background_water(cfg_bad_through)

        cfg_bad_normal = PhantomConfig(voxel_size_mm = 1.0,
                                       include_plates = [:water],
                                       slice_thickness_mm = 1.0,
                                       slice_normal = (0.0, 0.0, 0.0))
        @test_throws ErrorException build_phantom(cfg_bad_normal)
    end

    @testset "PD jitter respects coarse-water weights" begin
        cfg = PhantomConfig(voxel_size_mm = 1.0,
                            include_plates = [:water],
                            water_voxel_size_mm = 3.0,
                            slice_thickness_mm = 1.0,
                            slice_center_mm = (0.0, 0.0, PLATE_Z_MM.T1),
                            augment = AugmentConfig(PD_sigma_abs = 2.0),
                            rng_seed = 7)
        obj = build_phantom(cfg)
        weight = 9.0
        material_ρ = obj.ρ ./ weight
        @test all((0.0 .<= material_ρ) .& (material_ρ .<= 1.0))
        @test maximum(obj.ρ) <= weight + 1e-12
        @test any(obj.ρ .> 1.0)
    end
end
