using Suppressor: @suppress
using KomaMRI

# End-to-end smoke test: FID on a single sphere from the T1 array → the
# exponential decay envelope should recover the nominal T2 within a few %.

@testset "simulation smoke" begin
    cfg = PhantomConfig(voxel_size_mm = 2.0, include_plates = [:T1])
    descs = sphere_descriptors(:T1, cfg)
    d = descs[6]                                   # middle of the array
    obj = build_sphere(d, cfg.voxel_size_mm * 1e-3)
    @test length(obj.x) > 0

    # 90° pulse then ADC, same pattern as 01-FID.jl
    sys = Scanner()
    ampRF = 2e-6
    durRF = π / 2 / (2π * γ * ampRF)
    exc   = RF(ampRF, durRF)
    nADC, durADC = 2048, 4 * d.T2        # capture several T2 worth
    acq   = ADC(nADC, durADC, 1e-3)

    seq  = Sequence()
    seq += exc
    seq += acq

    raw = @suppress simulate(obj, seq, sys)

    # `raw` is a RawAcquisitionData; pull the complex samples.
    samples = raw.profiles[1].data[:, 1]
    @test length(samples) == nADC

    mag = abs.(samples)
    # Peak is near the beginning; signal decays monotonically in expectation.
    @test mag[1] > mag[end]

    # Fit A*exp(-t/T2) via least squares on log(mag).
    ts = range(0, durADC; length = nADC) .+ 1e-3
    # Restrict to the regime where signal is well above numerical noise
    keep = mag .> 0.01 * maximum(mag)
    @test sum(keep) > 10

    lm = log.(mag[keep])
    X  = hcat(ones(sum(keep)), -collect(ts[keep]))
    coefs  = X \ lm
    T2_fit = 1 / coefs[2]
    @test isapprox(T2_fit, d.T2; rtol = 0.10)
end
