# Exercises the PlotlyJS extension `plot_phantom_html`. In the direct
# `julia --project=. test/runtests.jl` workflow, KomaMRI currently loads PlotlyJS
# transitively and activates the extension. In `Pkg.test`, PlotlyJS is also listed
# as a test extra. HTML export needs no Kaleido, so this is CI/headless-safe.
using KomaMRI

@testset "plot_phantom_html" begin
    plot_ext = Base.get_extension(MRISystemPhantom, :MRISystemPhantomPlotlyJSExt)
    if plot_ext === nothing
        @test_skip "PlotlyJS extension is not loaded"
    else
        PlotlyJS = getfield(plot_ext, :PlotlyJS)

        @testset "full phantom: one trace per group" begin
            cfg = PhantomConfig(voxel_size_mm = 3.0, water_voxel_size_mm = 5.0)
            fig = plot_phantom_html(cfg)
            @test fig isa PlotlyJS.SyncPlot
            names = [t[:name] for t in fig.plot.data]
            # water + T1 + T2 + PD + fiducials
            @test Set(names) == Set(["water", "T1 plate", "T2 plate", "PD plate", "fiducials"])
            # water is drawn first (under the opaque spheres)
            @test names[1] == "water"
            # every sphere group has spins and uses the shared layout colour axis
            for t in fig.plot.data
                @test occursin("x: %{customdata[0]:.3f} mm", t[:hovertemplate])
                @test occursin("T1: %{customdata[3]:.5g} s", t[:hovertemplate])
                @test occursin("Δw: %{customdata[7]:.5g} Hz", t[:hovertemplate])
                @test length(t[:customdata]) == length(t[:x])
                @test length(first(t[:customdata])) == 8
                t[:name] == "water" && continue
                @test length(t[:x]) > 0
                @test t[:marker][:coloraxis] == "coloraxis"
            end
            @test fig.plot.layout[:coloraxis][:colorbar][:title][:text] == "T1"
            @test fig.plot.layout[:title][:text] ==
                  "T15 phantom - spheres coloured by T1"
            labels = [b[:label] for b in fig.plot.layout[:updatemenus][1][:buttons]]
            @test labels == ["T1", "T2", "T2s", "ρ", "Δw"]
            t2_button = fig.plot.layout[:updatemenus][1][:buttons][2]
            @test t2_button[:args][2]["title.text"] ==
                  "T15 phantom - spheres coloured by T2"
            dw_button = fig.plot.layout[:updatemenus][1][:buttons][5]
            @test dw_button[:args][2]["coloraxis.colorbar.title.text"] == "Δw"
            @test dw_button[:args][2]["title.text"] ==
                  "T15 phantom - spheres coloured by Δw"
        end

        @testset "no water when excluded" begin
            cfg = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:T1, :fiducials])
            fig = plot_phantom_html(cfg)
            names = [t[:name] for t in fig.plot.data]
            @test "water" ∉ names
            @test Set(names) == Set(["T1 plate", "fiducials"])
        end

        @testset "custom spheres become their own group" begin
            d = SphereDescriptor((0.0, 0.0, 0.0), 7.5e-3, 1.0, 0.5, 0.1, 0.1, 0.0, :c1)
            cfg = PhantomConfig(voxel_size_mm = 3.0, include_plates = [:water],
                                custom_sphere_descriptors = [d])
            fig = plot_phantom_html(cfg)
            names = [t[:name] for t in fig.plot.data]
            @test Set(names) == Set(["water", "custom spheres"])
        end

        @testset "HTML round-trips to disk" begin
            cfg = PhantomConfig(voxel_size_mm = 4.0, water_voxel_size_mm = 6.0)
            path = tempname() * ".html"
            try
                plot_phantom_html(cfg; file = path)
                @test isfile(path)
                @test filesize(path) > 0
            finally
                isfile(path) && rm(path)
            end
        end

        @testset "invalid opacity_sliders errors" begin
            @test_throws ErrorException plot_phantom_html(PhantomConfig(voxel_size_mm = 4.0);
                                                          opacity_sliders = :bogus)
        end
    end
end
