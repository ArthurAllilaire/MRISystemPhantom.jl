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

        @testset "water can join the property colour axis" begin
            cfg = PhantomConfig(voxel_size_mm = 3.0, water_voxel_size_mm = 5.0)
            fig = plot_phantom_html(cfg; color_water = true)
            names = [t[:name] for t in fig.plot.data]
            @test names[1] == "water"
            @test fig.plot.data[1][:marker][:coloraxis] == "coloraxis"
            @test length(fig.plot.data[1][:marker][:color]) == length(fig.plot.data[1][:x])
            @test fig.plot.layout[:title][:text] == "T15 phantom - spins coloured by T1"

            t2_button = fig.plot.layout[:updatemenus][1][:buttons][2]
            @test t2_button[:args][3] == collect(0:(length(fig.plot.data) - 1))
            @test length(t2_button[:args][1]["marker.color"]) == length(fig.plot.data)
            @test t2_button[:args][2]["title.text"] ==
                  "T15 phantom - spins coloured by T2"
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

        @testset "random phantom explorer: frames + property/opacity controls" begin
            rpcfg = RandomPhantomConfig(
                base = PhantomConfig(voxel_size_mm = 4.0,
                                     include_plates = [:T1, :T2, :PD, :water],
                                     water_voxel_size_mm = 6.0),
                sphere_selector  = SphereCountPerPlate(:T1 => 5, :T2 => 3, :PD => 2),
                material_sampler = RatioPreservingLogNormalT1(0.2),
                pose_sampler     = InPlanePoseSampler(rotation_sigma_rad = 0.1))
            props = (:T1, :T2, :T2s, :ρ, :Δw)
            fig = plot_random_phantom_explorer_html(rpcfg; seeds = 1:3, color_by = :T2)
            @test fig isa PlotlyJS.SyncPlot

            # water trace + one sphere trace per property; only color_by visible
            d = fig.plot.data
            @test d[1][:name] == "water"
            @test length(d) == 1 + length(props)
            @test [get(t, :visible, true) for t in d] ==
                  [true; [p === :T2 for p in props]]
            @test all(t -> t[:marker][:coloraxis] == "coloraxis", d[2:end])

            # one frame per seed, each updating every trace
            @test length(fig.plot.frames) == 3
            @test fig.plot.frames[1][:traces] == collect(0:length(props))

            # colour-by dropdown toggles which sphere trace is visible
            colour_menu = fig.plot.layout[:updatemenus][1]
            @test [b[:label] for b in colour_menu[:buttons]] ==
                  ["T1", "T2", "T2s", "ρ", "Δw"]
            t2s_btn = colour_menu[:buttons][3]
            @test t2s_btn[:args][1]["visible"] == [p === :T2s for p in props]
            @test t2s_btn[:args][3] == collect(1:length(props))
            @test t2s_btn[:args][2]["coloraxis.colorbar.title.text"] == "T2s"

            # Resample/Pause buttons
            @test [b[:label] for b in fig.plot.layout[:updatemenus][2][:buttons]] ==
                  ["▶ Resample", "⏸ Pause"]

            # two sliders: episode (one step per seed) + water opacity
            sliders = fig.plot.layout[:sliders]
            @test length(sliders) == 2
            @test sliders[1][:currentvalue][:prefix] == "episode (seed): "
            @test length(sliders[1][:steps]) == 3
            @test sliders[2][:currentvalue][:prefix] == "water opacity: "
            @test sliders[2][:steps][end][:args][1]["marker.opacity"] == 1.0
            @test sliders[2][:steps][end][:args][2] == [0]   # targets water trace

            @test fig.plot.layout[:coloraxis][:colorbar][:title][:text] == "T2"
            @test occursin("seed 1", fig.plot.layout[:title][:text])

            # static config box reports the sampler + base metadata
            box = fig.plot.layout[:annotations][1][:text]
            @test occursin("RandomPhantomConfig", box)
            @test occursin("SphereCountPerPlate", box)            # selector
            @test occursin("RatioPreservingLogNormalT1", box)     # material
            @test occursin("InPlanePoseSampler", box)             # pose
            @test occursin("base PhantomConfig", box)
            @test occursin("plates: T1, T2, PD, water", box)
            # box sits bottom-right and has a show/hide toggle
            ann = fig.plot.layout[:annotations][1]
            @test ann[:xanchor] == "right" && ann[:yanchor] == "bottom"
            toggle = fig.plot.layout[:updatemenus][3]
            @test [b[:label] for b in toggle[:buttons]] == ["ⓘ Info", "Hide"]
            @test toggle[:buttons][2][:args][1]["annotations[0].visible"] == false

            @test_throws ErrorException plot_random_phantom_explorer_html(rpcfg; seeds = Int[])
        end

        @testset "explorer HTML round-trips to disk" begin
            rpcfg = RandomPhantomConfig(
                base = PhantomConfig(voxel_size_mm = 4.0, include_plates = [:T1, :water],
                                     water_voxel_size_mm = 6.0),
                sphere_selector = SphereCountPerPlate(:T1 => 4))
            path = tempname() * ".html"
            try
                plot_random_phantom_explorer_html(rpcfg; seeds = 1:2, file = path)
                @test isfile(path)
                @test filesize(path) > 0
            finally
                isfile(path) && rm(path)
            end
        end
    end
end
