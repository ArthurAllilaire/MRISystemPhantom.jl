using Distributions

# Helper: build a material-sampling context like the internal pipeline does.
_mkctx(plate, idx, d; base = PhantomConfig()) =
    (; plate, index = idx, label = d.label, field = base.field,
       serial_number_class = base.serial_number_class, cfg = base)

_labels(descs) = sort([d.label for d in descs])

# Look up the nominal descriptor for a label in a truth record.
_nominal_for(truth, plate, label) =
    only(d for d in truth.descriptors_nominal[plate] if d.label === label)

@testset "random_phantom" begin
    FAST = PhantomConfig(voxel_size_mm = 4.0)   # coarse, fast builds

    @testset "defaults move contrast plates into custom descriptors" begin
        rp = RandomPhantomConfig(base = FAST)
        ep = sample_phantom_config(rp; rng_seed = 1)
        # contrast plates leave include_plates, fiducials + water stay
        @test :T1 ∉ ep.cfg.include_plates
        @test :T2 ∉ ep.cfg.include_plates
        @test :PD ∉ ep.cfg.include_plates
        @test :fiducials ∈ ep.cfg.include_plates
        @test :water ∈ ep.cfg.include_plates
        # 14 + 14 + 14 contrast spheres moved to custom descriptors
        @test length(ep.cfg.custom_sphere_descriptors) == 42
        @test length(ep.truth.active_labels) == 42
        # nominal == sampled when there is no material sampler
        nom = reduce(vcat, values(ep.truth.descriptors_nominal))
        @test _labels(ep.cfg.custom_sphere_descriptors) == _labels(nom)
    end

    @testset "determinism and seeds" begin
        rp = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0, include_plates = [:T1, :water]),
            material_sampler = RatioPreservingLogNormalT1(0.2),
            pose_sampler = InPlanePoseSampler(rotation_sigma_rad = 0.05,
                                              translation_sigma_mm = 2.0))
        a = sample_phantom_config(rp; rng_seed = 42)
        b = sample_phantom_config(rp; rng_seed = 42)
        c = sample_phantom_config(rp; rng_seed = 43)

        # identical seed -> identical materials and pose
        @test [d.T1 for d in a.cfg.custom_sphere_descriptors] ==
              [d.T1 for d in b.cfg.custom_sphere_descriptors]
        @test a.truth.rotation == b.truth.rotation
        # different seed -> different materials
        @test [d.T1 for d in a.cfg.custom_sphere_descriptors] !=
              [d.T1 for d in c.cfg.custom_sphere_descriptors]

        # build seed is derived, differs from episode seed, recorded in truth,
        # and written into the deterministic config
        @test a.truth.episode_seed == 42
        @test a.truth.build_seed != a.truth.episode_seed
        @test a.cfg.rng_seed == a.truth.build_seed
        @test a.cfg.rng_seed == b.cfg.rng_seed       # stable across calls
        @test a.cfg.rng_seed != c.cfg.rng_seed       # varies per episode

        # build_seed is a pure function of the episode seed: swapping the
        # samplers (which change how many RNG draws happen) must not change it.
        rp_other = RandomPhantomConfig(
            base = rp.base,
            material_sampler = (rng, d, ctx) -> (; T1 = d.T1 * (1 + 0.5 * randn(rng))),
            pose_sampler = GaussianEulerPose(rotation_sigma_rad = 0.3))
        @test sample_phantom_config(rp_other; rng_seed = 42).truth.build_seed ==
              a.truth.build_seed
    end

    @testset "sphere selection" begin
        rp_all = RandomPhantomConfig(base = FAST)
        ep = sample_phantom_config(rp_all; rng_seed = 1)
        @test length(ep.cfg.custom_sphere_descriptors) == 42   # nothing -> keep all

        # per-plate counts
        rp_cnt = RandomPhantomConfig(base = FAST,
            sphere_selector = SphereCountPerPlate(:T1 => 6, :T2 => 0, :PD => 3))
        e2 = sample_phantom_config(rp_cnt; rng_seed = 2)
        @test length(e2.truth.active_indices_by_plate[:T1]) == 6
        @test length(e2.truth.active_indices_by_plate[:T2]) == 0
        @test length(e2.truth.active_indices_by_plate[:PD]) == 3

        # integer pooled count across contrast plates
        rp_pool = RandomPhantomConfig(base = FAST, sphere_selector = 10)
        e3 = sample_phantom_config(rp_pool; rng_seed = 3)
        @test length(e3.cfg.custom_sphere_descriptors) == 10

        # base dropout must NOT compound with the selector
        rp_drop = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0,
                                 augment = AugmentConfig(drop_sphere_p = 0.9)),
            sphere_selector = SphereCountPerPlate(:T1 => 14))
        e4 = sample_phantom_config(rp_drop; rng_seed = 4)
        @test length(e4.truth.active_indices_by_plate[:T1]) == 14
    end

    @testset "E2 selector forces indices and drops other plates" begin
        rp = RandomPhantomConfig(base = FAST,
            sphere_selector = E2SphereSelector(subset_size = 5, forced_indices = [1, 14]))
        ep = sample_phantom_config(rp; rng_seed = 9)
        idx = ep.truth.active_indices_by_plate[:T1]
        @test length(idx) == 5
        @test 1 in idx && 14 in idx
        @test !haskey(ep.truth.active_indices_by_plate, :T2)
        @test all(d -> startswith(String(d.label), "T1_"), ep.cfg.custom_sphere_descriptors)
    end

    @testset "material result handling" begin
        d = sphere_descriptors(:T1, PhantomConfig())[4]

        # nothing -> unchanged
        rp_id = RandomPhantomConfig(base = FAST,
            material_sampler = (rng, dd, ctx) -> nothing)
        kept = sample_phantom_config(rp_id; rng_seed = 1).truth.descriptors_sampled[:T1][4]
        nomT1 = sphere_descriptors(:T1, PhantomConfig(voxel_size_mm = 4.0))[4].T1
        @test kept.T1 == nomT1

        # NamedTuple with T2 but no T2s -> T2s defaults to T2
        rp_nt = RandomPhantomConfig(base = FAST,
            material_sampler = (rng, dd, ctx) -> ctx.plate === :T1 ? (; T2 = 0.05) : nothing)
        s = sample_phantom_config(rp_nt; rng_seed = 1).truth.descriptors_sampled[:T1][1]
        @test s.T2 == 0.05
        @test s.T2s == 0.05

        # full SphereDescriptor return overrides geometry/label
        moved = SphereDescriptor((9.9, 9.9, 9.9), 1e-3, 0.5, 1.0, 0.1, 0.1, 0.0, :moved)
        rp_full = RandomPhantomConfig(base = PhantomConfig(voxel_size_mm = 4.0,
                                                           include_plates = [:T1]),
            material_sampler = (rng, dd, ctx) -> dd.label === Symbol("T1_1") ? moved : nothing)
        out = sample_phantom_config(rp_full; rng_seed = 1).truth.descriptors_sampled[:T1]
        @test any(d2 -> d2.label === :moved && d2.centre == (9.9, 9.9, 9.9), out)
    end

    @testset "material validation" begin
        # negative T1
        rp_neg = RandomPhantomConfig(base = PhantomConfig(voxel_size_mm = 4.0,
                                                          include_plates = [:T1]),
            material_sampler = (rng, d, ctx) -> (; T1 = -0.1))
        @test_throws ErrorException sample_phantom_config(rp_neg; rng_seed = 1)

        # T2s > T2 rejected by default
        rp_t2s = RandomPhantomConfig(base = PhantomConfig(voxel_size_mm = 4.0,
                                                          include_plates = [:T1]),
            material_sampler = (rng, d, ctx) -> (; T2 = 0.05, T2s = 0.5))
        err = try
            sample_phantom_config(rp_t2s; rng_seed = 1); nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("T2s", err)

        # ρ out of bounds rejected
        rp_rho = RandomPhantomConfig(base = PhantomConfig(voxel_size_mm = 4.0,
                                                          include_plates = [:T1]),
            material_sampler = (rng, d, ctx) -> (; ρ = 5.0))
        @test_throws ErrorException sample_phantom_config(rp_rho; rng_seed = 1)

        # relaxing the T2s rule allows it
        rp_relaxed = RandomPhantomConfig(base = PhantomConfig(voxel_size_mm = 4.0,
                                                              include_plates = [:T1]),
            enforce_t2s_le_t2 = false,
            material_sampler = (rng, d, ctx) -> (; T2 = 0.05, T2s = 0.5))
        @test sample_phantom_config(rp_relaxed; rng_seed = 1) isa RandomPhantomEpisode
    end

    @testset "pose sampling" begin
        # in-plane: only z-rotation, only x/y translation
        rp = RandomPhantomConfig(base = FAST,
            pose_sampler = InPlanePoseSampler(rotation_sigma_rad = 0.1,
                                              translation_sigma_mm = 3.0))
        ep = sample_phantom_config(rp; rng_seed = 5)
        @test ep.cfg.rotation[1] == 0.0 && ep.cfg.rotation[2] == 0.0
        @test ep.cfg.rotation[3] != 0.0
        @test ep.cfg.translation_mm[3] == 0.0

        # fixed pose keeps base pose by default
        base = PhantomConfig(voxel_size_mm = 4.0, rotation = (0.1, 0.2, 0.3))
        rpf = RandomPhantomConfig(base = base, pose_sampler = FixedPose())
        @test sample_phantom_config(rpf; rng_seed = 1).cfg.rotation == (0.1, 0.2, 0.3)
    end

    @testset "water cutout uses sampled custom descriptors" begin
        # Sampled spheres get a jittered T1 that differs from bulk water T1, so
        # T1 distinguishes a sphere spin from a water spin. If the water cutout
        # used the sampled custom descriptors, then every spin inside a sampled
        # sphere carries that sphere's T1 (no leftover water at the sphere site).
        rp = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0, include_plates = [:T1, :water]),
            material_sampler = RatioPreservingLogNormalT1(0.1))
        res = sample_phantom(rp; rng_seed = 1)
        water_T1 = BACKGROUND_WATER[res.cfg.field].T1
        @test length(res.phantom.x) > 0
        for d in res.cfg.custom_sphere_descriptors
            near = [i for i in eachindex(res.phantom.x)
                    if (res.phantom.x[i] - d.centre[1])^2 +
                       (res.phantom.y[i] - d.centre[2])^2 +
                       (res.phantom.z[i] - d.centre[3])^2 < (0.9 * d.radius)^2]
            @test !isempty(near)
            # every spin at the sphere site has the sampled T1, none is bulk water
            @test all(i -> isapprox(res.phantom.T1[i], d.T1; rtol = 1e-6), near)
            @test all(i -> !isapprox(res.phantom.T1[i], water_T1; rtol = 1e-6), near)
        end
    end

    @testset "declarative DSL — dependency ordering" begin
        # T1 plate: T1 sampled, T2 preserves nominal ratio (needs T1 first)
        d1 = sphere_descriptors(:T1, PhantomConfig())[4]
        s1 = MaterialDistributionSampler(
            T1 = PerPlate(:T1 => Uniform(0.4, 0.6)),
            T2 = PreserveNominalRatio(:T2, :T1))
        r1 = MRISystemPhantom._material_result(s1, MersenneTwister(0), d1, _mkctx(:T1, 4, d1))
        @test isapprox(r1[:T2] / r1[:T1], d1.T2 / d1.T1; rtol = 1e-9)

        # T2 plate: T2 sampled, T1 scaled from T2 (opposite order)
        d2 = sphere_descriptors(:T2, PhantomConfig())[3]
        s2 = MaterialDistributionSampler(
            T2 = PerPlate(:T2 => Uniform(0.02, 0.03)),
            T1 = ScaledFrom(:T2, 10.0))
        r2 = MRISystemPhantom._material_result(s2, MersenneTwister(0), d2, _mkctx(:T2, 3, d2))
        @test isapprox(r2[:T1], 10 * r2[:T2]; rtol = 1e-9)

        # cycle is detected
        sc = MaterialDistributionSampler(
            T1 = ScaledFrom(:T2, 2.0),
            T2 = PreserveNominalRatio(:T2, :T1))
        @test_throws ErrorException MRISystemPhantom._material_result(
            sc, MersenneTwister(0), d1, _mkctx(:T1, 4, d1))
    end

    @testset "declarative DSL — container lookup" begin
        t1 = sphere_descriptors(:T1, PhantomConfig())

        # PerSphere indexes by stable label index, invariant to subset selection
        ps = MaterialDistributionSampler(
            T1 = PerPlate(:T1 => PerSphere(Dict(4 => 0.111, 14 => 2.5); default = 1.0)))
        r4  = MRISystemPhantom._material_result(ps, MersenneTwister(0), t1[4],  _mkctx(:T1, 4,  t1[4]))
        r14 = MRISystemPhantom._material_result(ps, MersenneTwister(0), t1[14], _mkctx(:T1, 14, t1[14]))
        r5  = MRISystemPhantom._material_result(ps, MersenneTwister(0), t1[5],  _mkctx(:T1, 5,  t1[5]))
        @test r4[:T1]  == 0.111
        @test r14[:T1] == 2.5
        @test r5[:T1]  == 1.0      # falls through to default

        # PerLabel exact-label override beats nothing
        pl = MaterialDistributionSampler(T1 = PerLabel(Symbol("T1_2") => 0.42))
        @test MRISystemPhantom._material_result(pl, MersenneTwister(0), t1[2],
                  _mkctx(:T1, 2, t1[2]))[:T1] == 0.42
        @test MRISystemPhantom._material_result(pl, MersenneTwister(0), t1[3],
                  _mkctx(:T1, 3, t1[3])) === nothing   # no spec -> keep nominal

        # constant property function with full (rng, d, ctx) signature
        fn = MaterialDistributionSampler(T1 = (rng, d, ctx) -> d.T1 * 2)
        @test MRISystemPhantom._material_result(fn, MersenneTwister(0), t1[1],
                  _mkctx(:T1, 1, t1[1]))[:T1] == t1[1].T1 * 2
    end

    @testset "declarative DSL end-to-end through the builder" begin
        material = MaterialDistributionSampler(
            T1 = PerPlate(:T1 => Uniform(0.3, 2.5), :T2 => ScaledFrom(:T2, 10.0)),
            T2 = PerPlate(:T1 => PreserveNominalRatio(:T2, :T1),
                          :T2 => Uniform(0.02, 0.10)),
            ρ  = Truncated(Normal(1.0, 0.02), 0.0, 1.0))
        rp = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0, include_plates = [:T1, :T2, :water]),
            material_sampler = material)
        res = sample_phantom(rp; rng_seed = 11)
        @test length(res.phantom.x) > 0
        for d in res.truth.descriptors_sampled[:T1]
            nom = _nominal_for(res.truth, :T1, d.label)
            @test 0.3 <= d.T1 <= 2.5
            @test isapprox(d.T2 / d.T1, nom.T2 / nom.T1; rtol = 1e-9)
        end
        for d in res.truth.descriptors_sampled[:T2]
            @test 0.02 <= d.T2 <= 0.10
            @test isapprox(d.T1, 10 * d.T2; rtol = 1e-9)
        end
    end

    @testset "eval_episodes pool is reproducible and disjoint" begin
        rp = RandomPhantomConfig(base = FAST,
            material_sampler = RatioPreservingLogNormalT1(0.2))
        pool_a = eval_episodes(rp, 1:5)
        pool_b = eval_episodes(rp, 1:5)
        @test [e.truth.build_seed for e in pool_a] == [e.truth.build_seed for e in pool_b]
        @test [d.T1 for d in pool_a[1].cfg.custom_sphere_descriptors] ==
              [d.T1 for d in pool_b[1].cfg.custom_sphere_descriptors]
    end

    @testset "random pipeline owns the contrast plates it touches" begin
        # A selector that returns only :T1 must NOT leave :T2/:PD as deterministic
        # generated plates — they are candidates, so they leave include_plates.
        rp = RandomPhantomConfig(base = FAST,            # default 5 plates
            sphere_selector = E2SphereSelector(subset_size = 4))
        ep = sample_phantom_config(rp; rng_seed = 1)
        @test :T1 ∉ ep.cfg.include_plates
        @test :T2 ∉ ep.cfg.include_plates
        @test :PD ∉ ep.cfg.include_plates
        @test ep.cfg.include_plates == [:fiducials, :water]
        @test length(ep.cfg.custom_sphere_descriptors) == 4   # only sampled T1
        # built phantom contains no deterministic T2/PD spheres (would be ρ=1
        # contrast spheres off the T1 slab) — only the 4 sampled + water + fiducials
        @test all(d -> startswith(String(d.label), "T1_"),
                  ep.cfg.custom_sphere_descriptors)
    end

    @testset "build config has selection/dropout knobs cleared" begin
        rp = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0,
                augment = AugmentConfig(drop_sphere_p = 0.9, B0_sigma_Hz = 5.0),
                keep_sphere_labels = [:T1_1], drop_sphere_labels = [:T1_2]),
            sphere_selector = E2SphereSelector(subset_size = 4))
        ep = sample_phantom_config(rp; rng_seed = 1)
        @test ep.cfg.augment.drop_sphere_p == 0.0          # no build-time dropout
        @test ep.cfg.augment.B0_sigma_Hz == 5.0            # per-spin noise preserved
        @test ep.cfg.keep_sphere_labels === nothing
        @test isempty(ep.cfg.drop_sphere_labels)
    end

    @testset "pre-existing custom descriptors are preserved" begin
        extra = SphereDescriptor((0.05, 0.0, 0.0), 3e-3, 1.0, 0.3, 0.1, 0.1, 0.0, :extra_1)
        rp = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0, include_plates = [:T1],
                                 custom_sphere_descriptors = [extra]),
            sphere_selector = E2SphereSelector(subset_size = 2))
        ep = sample_phantom_config(rp; rng_seed = 1)
        @test any(d -> d.label === :extra_1, ep.cfg.custom_sphere_descriptors)
        @test length(ep.cfg.custom_sphere_descriptors) == 3   # 1 deterministic + 2 sampled
    end

    @testset "active identity comes from source, survives relabelling" begin
        relabel = (rng, d, ctx) ->
            SphereDescriptor(d.centre, d.radius, d.ρ, d.T1, d.T2, d.T2s, d.delta_w, :relabeled)
        rp = RandomPhantomConfig(
            base = PhantomConfig(voxel_size_mm = 4.0, include_plates = [:T1]),
            sphere_selector = E2SphereSelector(subset_size = 3, forced_indices = [5, 8, 11]),
            material_sampler = relabel)
        ep = sample_phantom_config(rp; rng_seed = 1)
        @test ep.truth.active_indices_by_plate[:T1] == [5, 8, 11]
        @test ep.truth.active_labels == [:T1_5, :T1_8, :T1_11]
        # the sampled descriptors themselves carry the new label
        @test all(d -> d.label === :relabeled, ep.truth.descriptors_sampled[:T1])
    end

    @testset "declarative DSL draw order follows _MATERIAL_PROPS" begin
        # Two independent properties (ρ then T1 in _MATERIAL_PROPS order): the
        # sampled values must match draws taken in that fixed order, independent of
        # Dict iteration. Repeated calls with the same seed are also identical.
        d = sphere_descriptors(:T1, PhantomConfig())[4]
        s = MaterialDistributionSampler(T1 = Uniform(0.4, 2.0), ρ = Uniform(0.5, 1.0))
        r1 = MRISystemPhantom._material_result(s, MersenneTwister(0), d, _mkctx(:T1, 4, d))
        r2 = MRISystemPhantom._material_result(s, MersenneTwister(0), d, _mkctx(:T1, 4, d))
        @test r1[:T1] == r2[:T1] && r1[:ρ] == r2[:ρ]       # reproducible
        rng = MersenneTwister(0)
        ρ_first = rand(rng, Uniform(0.5, 1.0))             # ρ drawn before T1
        T1_next = rand(rng, Uniform(0.4, 2.0))
        @test r1[:ρ]  == ρ_first
        @test r1[:T1] == T1_next
    end
end
