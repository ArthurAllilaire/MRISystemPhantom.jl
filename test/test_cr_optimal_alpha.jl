# Tests for the α-aware CR-optimal solver and Ernst-angle helpers
# (see ALPHA_DOF.md). The α-aware path lives in a separate function
# family (cr_optimal_alpha.jl); these tests pin that (a) it reduces exactly to
# the α=90° solver at α=π/2, (b) freeing α never worsens the objective, and
# (c) the Ernst helpers are correct.

using Test
using Random
using MRISystemPhantom

# Internal refinement helper (not exported), reached the same way
# test_e2_imaging.jl reaches _e2_simulate_step.
const _refine_alpha = MRISystemPhantom.refine_coordinate_descent_alpha

@testset "α-aware CR-optimal + Ernst" begin

    Npe = 8
    T1s = [1.398, 0.367, 0.064]                 # long / mid / short fleet sample
    TIs = [0.08, 0.4, 1.2, 2.0]
    TRs = [0.6,  1.0, 1.8, 2.8]
    n   = length(TIs)

    # At α=π/2 the α-aware per-sphere variance must equal the α=90° solver's
    # (the α path only changes where the 2×2 Jacobian is evaluated).
    @testset "α=π/2 reduces to the α=90° path (per-sphere variance)" begin
        αs = fill(π/2, n)
        for T1 in T1s
            v_alpha = cr_T1_variance_alpha(T1, TIs, TRs, αs; Npe = Npe)
            v_base  = cr_T1_variance(T1, TIs, TRs; Npe = Npe)
            @test isapprox(v_alpha, v_base; rtol = 1e-6)
        end
    end

    # Same reduction at the fleet-objective level (A-optimality), rtol 1e-6.
    @testset "α=π/2 reduces to the α=90° path (fleet objective)" begin
        αs = fill(π/2, n)
        L_alpha = cr_fleet_objective_alpha(T1s, TIs, TRs, αs; Npe = Npe)
        L_base  = cr_fleet_objective(T1s, TIs, TRs; Npe = Npe)
        @test isapprox(L_alpha, L_base; rtol = 1e-6)
        @test isfinite(L_alpha) && L_alpha > 0
    end

    # The 2×2 Fisher must stay well-posed away from π/2: Var(T1) positive and
    # finite across the fleet × {20°,50°,80°}.
    @testset "Fisher 2×2 well-posed: variance positive & finite" begin
        for T1 in T1s, αdeg in (20.0, 50.0, 80.0)
            αs = fill(deg2rad(αdeg), n)
            v = cr_T1_variance_alpha(T1, TIs, TRs, αs; Npe = Npe)
            @test isfinite(v) && v > 0
        end
    end

    # Coordinate descent only accepts improving moves → L1 ≤ L0.
    @testset "refinement never worsens the objective" begin
        αs0 = fill(deg2rad(60.0), n)
        L0  = cr_fleet_objective_alpha(T1s, TIs, TRs, αs0; Npe = Npe)
        _, _, _, L1 = _refine_alpha(
            TIs, TRs, αs0, T1s; Npe = Npe, budget_s = 1.0e6, n_iter = 20)
        @test L1 <= L0 + 1e-12
    end

    # α=π/2 is in the feasible window, so refining from the α=90° optimum can
    # only lower or equal the objective — α-freedom is never a regression.
    @testset "α-freedom never worse than the α=90° optimum" begin
        budget = 1.0e6
        base = cr_optimize(T1s; n_blocks = n, budget_s = budget, Npe = Npe,
                           n_starts = 1, n_refine = 1,
                           rng = MersenneTwister(0))
        # The α=90° optimum is reachable by the α-aware solver (α=π/2 ∈ window),
        # so refining from it can only lower (or equal) the objective.
        L90 = cr_fleet_objective(T1s, base.TIs, base.TRs; Npe = Npe)
        L90_alpha = cr_fleet_objective_alpha(T1s, base.TIs, base.TRs,
                                             fill(π/2, length(base.TIs)); Npe = Npe)
        @test isapprox(L90, L90_alpha; rtol = 1e-6)
        _, _, _, L_ref = _refine_alpha(
            base.TIs, base.TRs, fill(π/2, length(base.TIs)), T1s;
            Npe = Npe, budget_s = budget, n_iter = 30)
        @test L_ref <= L90_alpha + 1e-12
    end

    # Full solver smoke: finite L>0, right-length schedule, all αs in [5°,90°].
    @testset "cr_optimize_alpha end-to-end (smoke)" begin
        αlo, αhi = deg2rad(5.0), deg2rad(90.0)
        r = cr_optimize_alpha([1.0, 0.3]; n_blocks = 4, budget_s = 400.0,
                              Npe = Npe, n_starts = 100, n_refine = 5,
                              rng = MersenneTwister(1))
        @test isfinite(r.L) && r.L > 0
        @test length(r.TIs) == 4 && length(r.TRs) == 4 && length(r.αs) == 4
        @test all(αlo - 1e-9 .<= r.αs .<= αhi + 1e-9)
    end

    @testset "single-sphere: report optimized α vs Ernst (informative)" begin
        # Not a hard Ernst assertion — the CRLB-optimal α for T1 *estimation* is
        # related to but not identical to the SNR Ernst angle. We assert the
        # solver returns a sane in-window α and log it next to the Ernst value.
        r = cr_optimize_alpha([0.5]; n_blocks = 4, budget_s = 400.0, Npe = Npe,
                              n_starts = 200, n_refine = 8, rng = MersenneTwister(2))
        ernst = [rad2deg(ernst_angle(TR, 0.5)) for TR in r.TRs]
        @info "single-sphere α-opt" αs_deg=round.(rad2deg.(r.αs), digits=1) ernst_deg=round.(ernst, digits=1)
        @test all(deg2rad(5.0) - 1e-9 .<= r.αs .<= deg2rad(90.0) + 1e-9)
    end

    # ernst_angle is exactly acos(exp(-TR/T1)).
    @testset "ernst_angle matches acos(exp(-TR/T1))" begin
        for (TR, T1) in ((0.5, 1.0), (2.0, 0.3), (1.0, 0.064))
            @test isapprox(ernst_angle(TR, T1), acos(exp(-TR / T1)); rtol = 1e-12)
        end
    end

    @testset "ernst_fixed_schedule: one α/block, in bounds, right direction" begin
        αlo, αhi = deg2rad(5.0), deg2rad(90.0)
        αs = ernst_fixed_schedule(TIs, TRs, 0.4)
        @test length(αs) == n
        @test all(αlo - 1e-9 .<= αs .<= αhi + 1e-9)
        # Short TR + long T1 → small Ernst angle; long TR + short T1 → large.
        a_small = ernst_angle(0.3, 1.8)     # short TR, long T1
        a_large = ernst_angle(4.0, 0.05)    # long TR, short T1
        @test a_small < a_large
    end

end
