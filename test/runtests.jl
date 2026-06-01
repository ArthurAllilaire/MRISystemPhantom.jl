using Test
using Random
using MRISystemPhantom
using KomaMRI

@testset "MRISystemPhantom" begin
    include("test_materials.jl")
    include("test_geometry.jl")
    include("test_imaging.jl")
    include("test_forward_model.jl")
    include("test_builder.jl")
    include("test_augment.jl")
    include("test_determinism.jl")
    include("test_simulation.jl")
    include("test_baseline.jl")
    include("test_e1.jl")
    include("test_e2.jl")
    include("test_e2_imaging.jl")
    include("test_fit_alpha.jl")
    include("test_cr_optimal_alpha.jl")
    include("test_snr.jl")
end
