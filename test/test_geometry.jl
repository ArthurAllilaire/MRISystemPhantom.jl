# Minimal duck-typed phantom for occupancy tests — avoids a full KomaMRI build.
using LinearAlgebra: dot, norm

struct _MockPhantom
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
end

@testset "geometry primitives" begin
    @testset "voxelise_sphere volume" begin
        # At 1 mm voxels, voxelised volume should track (4/3)πr³ within 2 %
        # for a 15 mm diameter sphere (radius 7.5 mm).
        r = 7.5e-3
        delta_x = 1e-3
        xs, ys, zs = voxelise_sphere((0.0, 0.0, 0.0), r, delta_x)
        voxel_vol = length(xs) * delta_x^3
        @test isapprox(voxel_vol, sphere_volume(r); rtol = 0.02)

        # Coordinates are in metres and lie inside the sphere
        @test all(@.(xs^2 + ys^2 + zs^2) .<= r^2 + 1e-12)

        # Finer grid is more accurate
        delta_x2 = 0.5e-3
        xs2, _, _ = voxelise_sphere((0.0, 0.0, 0.0), r, delta_x2)
        voxel_vol2 = length(xs2) * delta_x2^3
        @test abs(voxel_vol2 - sphere_volume(r)) <
              abs(voxel_vol  - sphere_volume(r))

        # Non-centred sphere returns the same count (invariance under translation)
        xs3, _, _ = voxelise_sphere((0.1, -0.05, 0.03), r, delta_x)
        @test length(xs3) == length(xs)
    end

    @testset "oriented slice plane helpers" begin
        n, u, v = slice_basis((1.0, 2.0, 3.0))
        @test isapprox(norm(collect(n)), 1.0; atol = 1e-12)
        @test isapprox(norm(collect(u)), 1.0; atol = 1e-12)
        @test isapprox(norm(collect(v)), 1.0; atol = 1e-12)
        @test isapprox(dot(collect(n), collect(u)), 0.0; atol = 1e-12)
        @test isapprox(dot(collect(n), collect(v)), 0.0; atol = 1e-12)
        @test isapprox(dot(collect(u), collect(v)), 0.0; atol = 1e-12)
        @test_throws ErrorException slice_basis((0.0, 0.0, 0.0))

        centre = (0.0, 0.0, 0.0565)
        @test signed_distance(0.0, 0.0, 0.0570, centre, (0.0, 0.0, 1.0)) ≈ 0.5e-3

        oblique_centre = (0.02 * n[1], 0.02 * n[2], 0.02 * n[3])
        p_on_plane = (
            oblique_centre[1] + 0.01 * u[1] - 0.03 * v[1],
            oblique_centre[2] + 0.01 * u[2] - 0.03 * v[2],
            oblique_centre[3] + 0.01 * u[3] - 0.03 * v[3],
        )
        @test isapprox(signed_distance(p_on_plane..., oblique_centre, n), 0.0; atol = 1e-12)
    end

    @testset "voxelise_plane footprint" begin
        R = 0.100
        dx = 0.003
        centre = (0.0, 0.0, PLATE_Z_MM.T1 * 1e-3)
        n, u, v = slice_basis((0.0, 0.0, 1.0))
        xs, ys, zs = voxelise_plane(centre, n, u, v, dx, R)
        @test length(xs) > 0
        @test all(isapprox.(zs, centre[3]; atol = 1e-12))
        @test all(@.(xs^2 + ys^2 + zs^2) .<= R^2 + 1e-12)
        r_plane = sqrt(R^2 - centre[3]^2)
        @test maximum(sqrt.(xs.^2 .+ ys.^2)) <= r_plane + dx

        xs_empty, ys_empty, zs_empty = voxelise_plane((0.0, 0.0, 0.110), n, u, v, dx, R)
        @test isempty(xs_empty) && isempty(ys_empty) && isempty(zs_empty)

        n2, u2, v2 = slice_basis((1.0, 2.0, 3.0))
        centre2 = (0.02 * n2[1], 0.02 * n2[2], 0.02 * n2[3])
        xo, yo, zo = voxelise_plane(centre2, n2, u2, v2, dx, R)
        @test length(xo) > 0
        @test all(abs.(signed_distance.(xo, yo, zo, Ref(centre2), Ref(n2))) .< 1e-12)
        @test all(@.(xo^2 + yo^2 + zo^2) .<= R^2 + 1e-12)

        shifted_centre = (
            centre2[1] + dx * u2[1],
            centre2[2] + dx * u2[2],
            centre2[3] + dx * u2[3],
        )
        xa, ya, za = voxelise_plane(shifted_centre, n2, u2, v2, dx, R)
        anchor_dist = @. (xa - shifted_centre[1])^2 +
                         (ya - shifted_centre[2])^2 +
                         (za - shifted_centre[3])^2
        @test minimum(anchor_dist) < 1e-24
    end

    @testset "contrast plate layout" begin
        for z in (56.5, 16.5, -23.5)
            cs = contrast_plate_centres(z)
            @test length(cs) == 14

            # All at the plate's z
            @test all(c -> isapprox(c[3], z * 1e-3), cs)

            # Outer ring radius 65 mm, inner ring 28 mm (first 10 outer, last 4 inner)
            rings = [sqrt(c[1]^2 + c[2]^2) for c in cs]
            @test all(isapprox.(rings[1:10],  65e-3))
            @test all(isapprox.(rings[11:14], 28e-3))
        end
    end

    @testset "phantom_occupancy 2D" begin
        Npe, Nfe = 16, 32
        FOV = 0.2

        # Single spin at origin → DC pixel (centred indexing)
        p = _MockPhantom([0.0], [0.0], [0.0])
        occ = phantom_occupancy(p, Npe, Nfe, FOV)
        @test occ[Npe ÷ 2 + 1, Nfe ÷ 2 + 1] == 1.0
        @test sum(occ) == 1.0

        # Single spin at a known offset → correct pixel
        x0, y0 = FOV / 4, FOV / 8
        p2 = _MockPhantom([x0], [y0], [0.0])
        occ2 = phantom_occupancy(p2, Npe, Nfe, FOV)
        expected_ife = mod(round(Int, x0 * Nfe / FOV) + Nfe ÷ 2, Nfe) + 1
        expected_ipe = mod(round(Int, y0 * Npe / FOV) + Npe ÷ 2, Npe) + 1
        @test occ2[expected_ipe, expected_ife] == 1.0
        @test sum(occ2) == 1.0

        # Multiple spins accumulate correctly
        p3 = _MockPhantom([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        occ3 = phantom_occupancy(p3, Npe, Nfe, FOV)
        @test occ3[Npe ÷ 2 + 1, Nfe ÷ 2 + 1] == 2.0

        # Output shape is correct
        @test size(occ) == (Npe, Nfe)

        # Spin at FOV/2 wraps to pixel 1 (modular indexing)
        p4 = _MockPhantom([FOV / 2], [0.0], [0.0])
        occ4 = phantom_occupancy(p4, Npe, Nfe, FOV)
        @test sum(occ4) == 1.0   # exactly one spin landed somewhere
    end

    @testset "phantom_occupancy 3D" begin
        Npe, Nfe, Nsl = 8, 16, 4
        FOV_xy = 0.2
        FOV_z  = 0.1

        # Single spin at origin → DC voxel
        p = _MockPhantom([0.0], [0.0], [0.0])
        occ = phantom_occupancy(p, Npe, Nfe, Nsl, FOV_xy, FOV_z)
        @test occ[Npe ÷ 2 + 1, Nfe ÷ 2 + 1, Nsl ÷ 2 + 1] == 1.0
        @test sum(occ) == 1.0

        # Off-centre spin in all three axes
        x0, y0, z0 = FOV_xy / 4, -FOV_xy / 8, FOV_z / 4
        p2 = _MockPhantom([x0], [y0], [z0])
        occ2 = phantom_occupancy(p2, Npe, Nfe, Nsl, FOV_xy, FOV_z)
        exp_ife = mod(round(Int, x0 * Nfe / FOV_xy) + Nfe ÷ 2, Nfe) + 1
        exp_ipe = mod(round(Int, y0 * Npe / FOV_xy) + Npe ÷ 2, Npe) + 1
        exp_isl = mod(round(Int, z0 * Nsl / FOV_z)  + Nsl ÷ 2, Nsl) + 1
        @test occ2[exp_ipe, exp_ife, exp_isl] == 1.0
        @test sum(occ2) == 1.0

        # Output shape is correct
        @test size(occ) == (Npe, Nfe, Nsl)
    end

    @testset "fiducial grid" begin
        pts = fiducial_grid_centres()
        @test length(pts) == 57

        # Each point is within the clipping sphere
        rmax = 95e-3
        @test all(p -> p[1]^2 + p[2]^2 + p[3]^2 <= rmax^2 + 1e-12, pts)

        # 40 mm spacing — every coordinate is a multiple of 0.04 m
        for p in pts
            for ci in p
                @test isapprox(round(ci / 0.04) * 0.04, ci; atol = 1e-12)
            end
        end
    end
end
