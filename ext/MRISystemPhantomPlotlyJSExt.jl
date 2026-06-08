# Package extension: interactive 3D phantom viewer.
#
# Loaded automatically once both MRISystemPhantom and PlotlyJS are present in the
# session. Implements `MRISystemPhantom.plot_phantom_html` (declared as a stub in
# the main module) so PlotlyJS stays a *weak* dependency — the core package never
# hard-depends on a plotting backend.
module MRISystemPhantomPlotlyJSExt

using PlotlyJS
import MRISystemPhantom: plot_phantom_html
using MRISystemPhantom: PhantomConfig, SphereDescriptor, build_phantom,
                        sphere_descriptors, transform_descriptor
using MRISystemPhantom: _mm_to_m

# Friendly legend/title names for each generated plate.
const _PLATE_LABEL = Dict(:T1 => "T1 plate", :T2 => "T2 plate",
                          :PD => "PD plate", :fiducials => "fiducials")

# Even-spacing decimation (mirrors examples/plot_random_phantom_3d.jl).
_subsample(idx, maxn) = length(idx) <= maxn ? idx :
    unique(idx[round.(Int, range(1, length(idx); length = maxn))])

_plot_title(cfg::PhantomConfig, color_by::Symbol, color_water::Bool) =
    "$(cfg.field) phantom - $(color_water ? "spins" : "spheres") coloured by $(color_by)"

function _hover_data(obj, idx)
    [[1e3 * obj.x[i], 1e3 * obj.y[i], 1e3 * obj.z[i],
      obj.T1[i], obj.T2[i], obj.T2s[i], obj.ρ[i], obj.Δw[i]] for i in idx]
end

const _HOVER_TEMPLATE =
    "<b>%{fullData.name}</b><br>" *
    "x: %{customdata[0]:.3f} mm<br>" *
    "y: %{customdata[1]:.3f} mm<br>" *
    "z: %{customdata[2]:.3f} mm<br>" *
    "T1: %{customdata[3]:.5g} s<br>" *
    "T2: %{customdata[4]:.5g} s<br>" *
    "T2s: %{customdata[5]:.5g} s<br>" *
    "ρ: %{customdata[6]:.5g}<br>" *
    "Δw: %{customdata[7]:.5g} Hz" *
    "<extra></extra>"

# Boolean mask: spins lying inside any of `descs`.
function _group_mask(obj, descs)
    m = falses(length(obj.x))
    for d in descs
        cx, cy, cz = d.centre
        r2 = d.radius^2
        @. m |= (obj.x - cx)^2 + (obj.y - cy)^2 + (obj.z - cz)^2 <= r2
    end
    m
end

function plot_phantom_html(cfg::PhantomConfig = PhantomConfig();
        color_by::Symbol = :T1,
        properties = (:T1, :T2, :T2s, :ρ, :Δw),
        max_water_points::Int = 200_000,
        max_sphere_points::Int = 400_000,
        color_water::Bool = false,
        water_opacity::Float64 = 0.08,
        opacity_sliders::Symbol = :water,
        height::Int = 720,
        file::Union{Nothing,AbstractString} = nothing)

    opacity_sliders in (:water, :all, :none) ||
        error("opacity_sliders must be :water, :all or :none (got $opacity_sliders)")

    obj = build_phantom(cfg)
    n = length(obj.x)
    n == 0 && error("phantom has no spins to plot")

    # --- group the spins by plate -----------------------------------------
    # Pose-correct descriptor centres so the geometric test matches the
    # (already rotated/translated) spins in `obj`.
    translation = _mm_to_m.(cfg.translation_mm)
    sphere_groups = Tuple{String,Vector{SphereDescriptor}}[]
    for plate in (:T1, :T2, :PD, :fiducials)
        plate in cfg.include_plates || continue
        descs = sphere_descriptors(plate, cfg)
        isempty(descs) && continue
        push!(sphere_groups, (_PLATE_LABEL[plate],
            [transform_descriptor(d, cfg.rotation, translation) for d in descs]))
    end
    if !isempty(cfg.custom_sphere_descriptors)
        push!(sphere_groups, ("custom spheres",
            [transform_descriptor(d, cfg.rotation, translation)
             for d in cfg.custom_sphere_descriptors]))
    end

    assigned = falses(n)
    group_idx = Vector{Vector{Int}}()       # subsampled spin indices per sphere group
    for (_, descs) in sphere_groups
        m = _group_mask(obj, descs)
        m .&= .!assigned                      # spheres never overlap, but be safe
        assigned .|= m
        push!(group_idx, _subsample(findall(m), max_sphere_points))
    end
    water_all = findall(.!assigned)
    water_idx = _subsample(water_all, max_water_points)

    # --- colour data (precompute every property for the live dropdown) -----
    sphere_all = findall(assigned)
    prop_vals = Dict(p => getproperty(obj, p) for p in properties)
    colour_range_idx = color_water ? vcat(sphere_all, water_all) : sphere_all
    # Shared colour range per property across all coloured spins, so groups are
    # comparable on one colorbar. By default water stays a constant translucent
    # trace so it does not stretch the sphere contrast range.
    prop_range = Dict(p => (isempty(colour_range_idx) ? (0.0, 1.0) :
                            extrema(prop_vals[p][colour_range_idx])) for p in properties)
    color_by in properties || (color_by = first(properties))

    # --- traces (water first so opaque spheres render on top) --------------
    traces = PlotlyJS.GenericTrace[]
    sphere_trace_idx = Int[]                  # 0-based Plotly indices of sphere traces
    if !isempty(water_idx)
        water_marker = color_water ?
            PlotlyJS.attr(size = 1.5, color = prop_vals[color_by][water_idx],
                          coloraxis = "coloraxis", opacity = water_opacity) :
            PlotlyJS.attr(size = 1.5, color = "lightblue",
                          opacity = water_opacity)
        push!(traces, PlotlyJS.scatter3d(
            x = obj.x[water_idx], y = obj.y[water_idx], z = obj.z[water_idx],
            mode = "markers", name = "water",
            customdata = _hover_data(obj, water_idx),
            hovertemplate = _HOVER_TEMPLATE,
            marker = water_marker))
    end
    cmin0, cmax0 = prop_range[color_by]
    for (gi, (name, _)) in enumerate(sphere_groups)
        idx = group_idx[gi]
        push!(sphere_trace_idx, length(traces))           # 0-based index of this trace
        push!(traces, PlotlyJS.scatter3d(
            x = obj.x[idx], y = obj.y[idx], z = obj.z[idx],
            mode = "markers", name = name,
            customdata = _hover_data(obj, idx),
            hovertemplate = _HOVER_TEMPLATE,
            marker = PlotlyJS.attr(size = 2.5, color = prop_vals[color_by][idx],
                coloraxis = "coloraxis")))
    end

    # --- live property dropdown (recolour sphere traces) -------------------
    updatemenus = []
    colour_trace_idx = copy(sphere_trace_idx)
    colour_group_idx = copy(group_idx)
    if color_water && !isempty(water_idx)
        colour_trace_idx = vcat(0, colour_trace_idx)
        colour_group_idx = vcat([water_idx], colour_group_idx)
    end
    if !isempty(colour_trace_idx) && length(properties) > 1
        buttons = [PlotlyJS.attr(label = String(p), method = "update",
            args = [Dict{String,Any}(
                "marker.color" => [prop_vals[p][idx] for idx in colour_group_idx]),
                Dict{String,Any}(
                    "coloraxis.cmin" => prop_range[p][1],
                    "coloraxis.cmax" => prop_range[p][2],
                    "coloraxis.colorbar.title.text" => String(p),
                    "title.text" => _plot_title(cfg, p, color_water)),
                colour_trace_idx])
            for p in properties]
        push!(updatemenus, PlotlyJS.attr(type = "dropdown", direction = "down",
            showactive = true, x = 0.0, y = 1.06, xanchor = "left", yanchor = "top",
            pad = PlotlyJS.attr(t = 4, l = 4),
            active = something(findfirst(==(color_by), properties), 1) - 1,
            buttons = buttons))
    end

    # --- opacity slider(s) -------------------------------------------------
    opacity_levels = round.(range(0.0, 1.0; length = 21); digits = 3)
    nearest(o) = argmin(abs.(opacity_levels .- o)) - 1     # 0-based active step
    sliders = []
    function _opacity_slider(trace_index, prefix, init, y)
        steps = [PlotlyJS.attr(label = string(o), method = "restyle",
                    args = [Dict{String,Any}("marker.opacity" => o), [trace_index]])
                 for o in opacity_levels]
        PlotlyJS.attr(active = nearest(init), x = 0.0, y = y, len = 0.5,
            pad = PlotlyJS.attr(t = 30, b = 10),
            currentvalue = PlotlyJS.attr(prefix = prefix, visible = true),
            steps = steps)
    end
    if opacity_sliders != :none && !isempty(water_idx)
        push!(sliders, _opacity_slider(0, "water opacity: ", water_opacity, -0.02))
    end
    if opacity_sliders == :all
        y = -0.02 - (isempty(water_idx) ? 0.0 : 0.10)
        for (gi, ti) in enumerate(sphere_trace_idx)
            push!(sliders, _opacity_slider(ti,
                "$(sphere_groups[gi][1]) opacity: ", 1.0, y))
            y -= 0.10
        end
    end

    bottom_margin = 50 + 60 * length(sliders)
    title = _plot_title(cfg, color_by, color_water)

    layout = PlotlyJS.Layout(height = height,
        title = PlotlyJS.attr(text = title, x = 0.5, xanchor = "center",
                              y = 0.98, yanchor = "top"),
        margin = PlotlyJS.attr(t = 110, b = bottom_margin, l = 10, r = 10),
        scene = PlotlyJS.attr(aspectmode = "data"),
        coloraxis = PlotlyJS.attr(colorscale = "Viridis", cmin = cmin0, cmax = cmax0,
            colorbar = PlotlyJS.attr(title = PlotlyJS.attr(text = String(color_by),
                                                           side = "right"))),
        legend = PlotlyJS.attr(x = 1.0, y = 1.0, xanchor = "right", yanchor = "top"),
        updatemenus = updatemenus, sliders = sliders)

    fig = PlotlyJS.plot(traces, layout)
    file !== nothing && PlotlyJS.savefig(fig, file; format = "html")
    fig
end

end # module
