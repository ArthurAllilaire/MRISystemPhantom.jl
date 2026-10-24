using KomaMRI: Scanner

@testset "material tables" begin
    # Table lengths — all 14 contrast spheres per array
    @test length(T1_ARRAY[:T15]) == 14
    @test length(T1_ARRAY[:T3])  == 14
    @test length(T2_OF_T1_ARRAY[:T15]) == 14
    @test length(T2_OF_T1_ARRAY[:T3])  == 14
    @test length(T2_ARRAY[:T15]) == 14
    @test length(T2_ARRAY[:T3])  == 14
    @test length(PD_FRACTIONS)   == 14

    # Spot-check manual values (4 sig fig) at 3 T
    @test T1_ARRAY[:T3][1]         == 1.838
    @test T1_ARRAY[:T3][14]        == 0.02295
    @test T2_OF_T1_ARRAY[:T3][7]   == 0.1893
    @test T2_ARRAY[:T3][1]         == 0.6458
    @test T2_ARRAY[:T3][14]        == 0.00535

    # And at 1.5 T
    @test T1_ARRAY[:T15][5]        == 0.527
    @test T2_ARRAY[:T15][10]       == 0.02728

    # PD is monotonically increasing; endpoints match manual table
    @test PD_FRACTIONS[1]  == 0.05
    @test PD_FRACTIONS[14] == 1.00
    @test issorted(PD_FRACTIONS)

    # T1-array values are monotonically decreasing (doping increases → T1 falls)
    @test issorted(T1_ARRAY[:T3];  rev = true)
    @test issorted(T1_ARRAY[:T15]; rev = true)
    @test issorted(T2_ARRAY[:T3];  rev = true)
    @test issorted(T2_ARRAY[:T15]; rev = true)

    # T2 < T1 on both arrays (physical sanity)
    @test all(T2_OF_T1_ARRAY[:T3]  .< T1_ARRAY[:T3])
    @test all(T2_OF_T1_ARRAY[:T15] .< T1_ARRAY[:T15])
    @test all(T2_ARRAY[:T3]  .< T1_OF_T2_ARRAY[:T3])
    @test all(T2_ARRAY[:T15] .< T1_OF_T2_ARRAY[:T15])

    # Background water sane
    @test BACKGROUND_WATER[:T3].T1  > 1.0
    @test BACKGROUND_WATER[:T15].T1 > 1.0
    @test BACKGROUND_WATER[:T3].ρ   == 1.0

    # Fiducial properties defined for both fields
    @test haskey(FIDUCIAL_PROPS, :T15)
    @test haskey(FIDUCIAL_PROPS, :T3)
    @test FIDUCIAL_PROPS[:T3].ρ == 1.0

    # Legacy T1 table is populated
    @test length(T1_ARRAY_LEGACY[:T3]) == 14
end

# A second, independent transcription of the manual's Table 1 (serial >= 0042), in ms,
# so a typo in one copy fails instead of silently changing the twin.
@testset "material tables match manual Table 1" begin
    manual_ms = Dict(
        :T15 => (
            T1_of_T1 = [1879, 1432, 1027, 751.3, 527.0, 384.1, 272.3, 194.5, 137.8, 94.7, 67.0, 48.14, 34.35, 24.16],
            T2_of_T1 = [1542, 1196, 871.7, 646.1, 457.5, 335.3, 238.4, 170.6, 121.6, 83.7, 59.1, 42.6, 30.4, 21.3],
            T1_of_T2 = [2640, 2292, 1923, 1489, 1245, 1004, 733.9, 533.1, 400.3, 261.0, 189.8, 154.7, 102.1, 79.65],
            T2_of_T2 = [1044, 623.9, 428.3, 258.4, 186.1, 137.0, 89.52, 62.82, 43.84, 27.28, 19.24, 15.44, 10.05, 7.79],
        ),
        :T3 => (
            T1_of_T1 = [1838, 1398, 998.3, 725.8, 509.1, 367.0, 258.7, 184.7, 130.8, 90.9, 64.2, 46.28, 32.65, 22.95],
            T2_of_T1 = [1354, 1035, 728.3, 524.4, 368.6, 266.7, 189.3, 134.1, 93.8, 65.7, 46.8, 33.11, 23.69, 16.73],
            T1_of_T2 = [2756, 2281, 1961, 1552, 1341, 1017, 782.1, 589.7, 443.8, 299.8, 237.8, 170.5, 121.8, 86.9],
            T2_of_T2 = [645.8, 423.6, 286.0, 184.8, 134.1, 94.40, 62.51, 44.98, 30.95, 20.10, 15.40, 10.85, 7.59, 5.35],
        ),
    )
    for (field, m) in manual_ms
        @test T1_ARRAY[field]       .* 1000 ≈ m.T1_of_T1
        @test T2_OF_T1_ARRAY[field] .* 1000 ≈ m.T2_of_T1
        @test T1_OF_T2_ARRAY[field] .* 1000 ≈ m.T1_of_T2
        @test T2_ARRAY[field]       .* 1000 ≈ m.T2_of_T2
    end
end

@testset "scanner_for_field" begin
    sys_t3  = scanner_for_field(:T3)
    sys_t15 = scanner_for_field(:T15)

    @test sys_t3.B0  == 3.0
    @test sys_t15.B0 == 1.5

    # PhantomConfig overload
    @test scanner_for_field(PhantomConfig(field = :T3)).B0  == 3.0
    @test scanner_for_field(PhantomConfig(field = :T15)).B0 == 1.5

    # Non-B0 fields are default Scanner values
    @test sys_t3.Gmax  == Scanner().Gmax
    @test sys_t15.Gmax == Scanner().Gmax
end
