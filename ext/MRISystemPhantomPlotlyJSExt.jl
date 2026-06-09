# Package extension: interactive 3D phantom viewer.
#
# Loaded automatically once both MRISystemPhantom and PlotlyJS are present in the
# session. Implements `MRISystemPhantom.plot_phantom_html` (declared as a stub in
# the main module) so PlotlyJS stays a *weak* dependency — the core package never
# hard-depends on a plotting backend.
module MRISystemPhantomPlotlyJSExt

using PlotlyJS
import MRISystemPhantom: plot_phantom_html, plot_random_phantom_explorer_html
using MRISystemPhantom: PhantomConfig, SphereDescriptor, build_phantom,
                        sphere_descriptors, transform_descriptor,
                        RandomPhantomConfig, sample_phantom_config
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

# ---------------------------------------------------------------------------
# Random phantom explorer: pre-sampled episodes baked into one animated figure.
# ---------------------------------------------------------------------------

# All pose-corrected sphere descriptors for a sampled episode (generated plates +
# the randomised custom spheres), so spins can be split into spheres vs water.
function _episode_descriptors(cfg::PhantomConfig)
    translation = _mm_to_m.(cfg.translation_mm)
    descs = SphereDescriptor[]
    for plate in (:T1, :T2, :PD, :fiducials)
        plate in cfg.include_plates || continue
        append!(descs, sphere_descriptors(plate, cfg))
    end
    append!(descs, cfg.custom_sphere_descriptors)
    [transform_descriptor(d, cfg.rotation, translation) for d in descs]
end

# Split a built episode into subsampled sphere / water spin indices.
function _episode_groups(obj, cfg, max_sphere_points, max_water_points)
    descs = _episode_descriptors(cfg)
    m = isempty(descs) ? falses(length(obj.x)) : _group_mask(obj, descs)
    (_subsample(findall(m), max_sphere_points),
     _subsample(findall(.!m), max_water_points))
end

# Human-readable rotation summary for the per-episode title.
_rotation_summary(r::NTuple{3,<:Real}) =
    "Euler° " * join(string.(round.(rad2deg.(r); digits = 1)), ", ")
_rotation_summary(::AbstractMatrix) = "SO(3) matrix"

_explorer_title(seed, n_spheres, rotation) =
    "random phantom — seed $seed — $n_spheres spheres — $(_rotation_summary(rotation))"

# Human-readable one-liner for a sampler / config value shown in the info box.
_short(s::AbstractString, n = 80) = length(s) > n ? first(s, n) * "…" : s
_describe(::Nothing)  = "none"
_describe(f::Function) = "⟨custom function⟩"
function _describe(x)
    fns = fieldnames(typeof(x))
    name = string(nameof(typeof(x)))
    isempty(fns) && return name
    name * "(" * join(("$f=$(_short(string(getfield(x, f))))" for f in fns), ", ") * ")"
end

# Static reference box: the RandomPhantomConfig knobs + base PhantomConfig
# metadata, so the distribution behind the resampling can be read straight off
# the figure. These do not change per episode (they are the distribution, not the
# draw), so the box is a single fixed annotation.
function _config_box_text(rpcfg::RandomPhantomConfig, seeds)
    b   = rpcfg.base
    aug = b.augment
    augnz = [(f, getfield(aug, f)) for f in fieldnames(typeof(aug)) if getfield(aug, f) != 0]
    augstr = isempty(augnz) ? "none" : join(("$f=$v" for (f, v) in augnz), ", ")
    water = b.water_voxel_size_mm === nothing ? "" :
        "   water voxel: $(b.water_voxel_size_mm) mm"
    lines = String[
        "<b>RandomPhantomConfig</b>",
        "seeds:     $(first(seeds))…$(last(seeds))  (n=$(length(seeds)))",
        "selector:  $(_describe(rpcfg.sphere_selector))",
        "material:  $(_describe(rpcfg.material_sampler))",
        "pose:      $(_describe(rpcfg.pose_sampler))",
        "ρ∈$(rpcfg.rho_bounds)   T2s≤T2: $(rpcfg.enforce_t2s_le_t2)   validate: $(rpcfg.validate)",
        "",
        "<b>base PhantomConfig</b>",
        "field: $(b.field)   voxel: $(b.voxel_size_mm) mm$(water)",
        "plates: $(join(string.(b.include_plates), ", "))",
        "augment: $(augstr)",
    ]
    if b.slice_thickness_mm !== nothing
        push!(lines, "slice: thick $(b.slice_thickness_mm) mm  normal $(b.slice_normal)")
    end
    join(lines, "<br>")
end

# One scatter3d per property (same sphere geometry, different baked colour), plus
# the translucent water trace. Only `color_by`'s sphere trace starts visible; the
# colour-by dropdown toggles `visible` — which animation frames never touch, so it
# composes with episode stepping. Frames update geometry + colour on every trace.
_water_trace(obj, w_idx, water_opacity) = PlotlyJS.scatter3d(
    x = obj.x[w_idx], y = obj.y[w_idx], z = obj.z[w_idx],
    mode = "markers", name = "water",
    customdata = _hover_data(obj, w_idx), hovertemplate = _HOVER_TEMPLATE,
    marker = PlotlyJS.attr(size = 1.5, color = "lightblue", opacity = water_opacity))

_sphere_trace(obj, s_idx, p, sphere_size, visible) = PlotlyJS.scatter3d(
    x = obj.x[s_idx], y = obj.y[s_idx], z = obj.z[s_idx],
    mode = "markers", name = "spheres ($(p))", showlegend = false, visible = visible,
    customdata = _hover_data(obj, s_idx), hovertemplate = _HOVER_TEMPLATE,
    marker = PlotlyJS.attr(size = sphere_size,
                           color = getproperty(obj, p)[s_idx], coloraxis = "coloraxis"))

function plot_random_phantom_explorer_html(rpcfg::RandomPhantomConfig;
        seeds = 1:10,
        properties = (:T1, :T2, :T2s, :ρ, :Δw),
        color_by::Symbol = :T1,
        # Caps are per-episode and every episode is baked into the file, so keep
        # them well below the single-figure viewer to keep the HTML small/snappy.
        max_water_points::Int = 12_000,
        max_sphere_points::Int = 40_000,
        water_opacity::Float64 = 0.12,
        sphere_size::Float64 = 2.6,
        height::Int = 720,
        play_ms::Int = 700,
        file::Union{Nothing,AbstractString} = nothing)

    seeds = collect(seeds)
    isempty(seeds) && error("`seeds` must be non-empty")
    color_by in properties || (color_by = first(properties))

    # --- pre-sample every episode -----------------------------------------
    episodes = map(seeds) do s
        ep  = sample_phantom_config(rpcfg; rng_seed = s)
        obj = build_phantom(ep.cfg)
        s_idx, w_idx = _episode_groups(obj, ep.cfg, max_sphere_points, max_water_points)
        (; ep, obj, s_idx, w_idx)
    end

    # Per-property colour range shared across episodes, so a colour means the same
    # thing in every frame and the dropdown can reset the axis when switching.
    prop_range = Dict(p => begin
        vals = reduce(vcat, [getproperty(e.obj, p)[e.s_idx] for e in episodes];
                      init = Float64[])
        isempty(vals) ? (0.0, 1.0) : extrema(vals)
    end for p in properties)

    titlefor(e) = _explorer_title(e.ep.truth.episode_seed,
        length(e.ep.cfg.custom_sphere_descriptors), e.ep.cfg.rotation)

    # --- base traces (first episode): water + one sphere trace per property ---
    e1 = episodes[1]
    traces = PlotlyJS.GenericTrace[_water_trace(e1.obj, e1.w_idx, water_opacity)]
    for p in properties
        push!(traces, _sphere_trace(e1.obj, e1.s_idx, p, sphere_size, p === color_by))
    end

    # --- one animation frame per episode (updates every trace) ------------
    frames = PlotlyJS.PlotlyFrame[]
    for (i, e) in enumerate(episodes)
        data = PlotlyJS.GenericTrace[_water_trace(e.obj, e.w_idx, water_opacity)]
        for p in properties
            push!(data, _sphere_trace(e.obj, e.s_idx, p, sphere_size, p === color_by))
        end
        push!(frames, PlotlyJS.frame(name = string(i), data = data,
            traces = collect(0:length(properties)),
            layout = PlotlyJS.attr(title = PlotlyJS.attr(text = titlefor(e)))))
    end

    # --- colour-by dropdown (toggle which property's sphere trace is visible) --
    sphere_trace_idx = collect(1:length(properties))    # 0-based: water is 0
    colour_buttons = [PlotlyJS.attr(label = String(p), method = "update",
        args = [Dict{String,Any}("visible" => [q === p for q in properties]),
                Dict{String,Any}(
                    "coloraxis.cmin" => prop_range[p][1],
                    "coloraxis.cmax" => prop_range[p][2],
                    "coloraxis.colorbar.title.text" => String(p)),
                sphere_trace_idx])
        for p in properties]
    colour_menu = PlotlyJS.attr(type = "dropdown", direction = "down", showactive = true,
        x = 0.0, y = 1.08, xanchor = "left", yanchor = "top",
        pad = PlotlyJS.attr(t = 4, l = 4),
        active = something(findfirst(==(color_by), properties), 1) - 1,
        buttons = colour_buttons)

    # --- episode slider (drag = forward/back) + Play/Pause ----------------
    anim_opts(dur) = PlotlyJS.attr(mode = "immediate", fromcurrent = true,
        frame = PlotlyJS.attr(duration = dur, redraw = true),
        transition = PlotlyJS.attr(duration = 0))
    steps = [PlotlyJS.attr(label = string(seeds[i]), method = "animate",
                args = [[string(i)], anim_opts(0)]) for i in eachindex(episodes)]
    episode_slider = PlotlyJS.attr(active = 0, x = 0.0, y = -0.06, len = 1.0,
        pad = PlotlyJS.attr(t = 20, b = 10),
        currentvalue = PlotlyJS.attr(prefix = "episode (seed): ", visible = true),
        steps = steps)
    playmenu = PlotlyJS.attr(type = "buttons", direction = "left", showactive = false,
        x = 0.30, y = 1.08, xanchor = "left", yanchor = "top",
        pad = PlotlyJS.attr(t = 4, r = 8),
        buttons = [
            PlotlyJS.attr(label = "▶ Resample", method = "animate",
                args = [nothing, anim_opts(play_ms)]),
            PlotlyJS.attr(label = "⏸ Pause", method = "animate",
                args = [[nothing], anim_opts(0)])])

    # --- water opacity slider (restyle persists across frames) ------------
    opacity_levels = round.(range(0.0, 1.0; length = 21); digits = 3)
    nearest(o) = argmin(abs.(opacity_levels .- o)) - 1
    opacity_steps = [PlotlyJS.attr(label = string(o), method = "restyle",
            args = [Dict{String,Any}("marker.opacity" => o), [0]])
        for o in opacity_levels]
    water_slider = PlotlyJS.attr(active = nearest(water_opacity), x = 0.0, y = -0.30,
        len = 0.5, pad = PlotlyJS.attr(t = 20, b = 10),
        currentvalue = PlotlyJS.attr(prefix = "water opacity: ", visible = true),
        steps = opacity_steps)

    # --- static reference box: the distribution behind the resampling -----
    config_box = PlotlyJS.attr(text = _config_box_text(rpcfg, seeds), visible = true,
        xref = "paper", yref = "paper", x = 1.0, y = 0.0,
        xanchor = "right", yanchor = "bottom", align = "left", showarrow = false,
        bordercolor = "#888", borderwidth = 1, borderpad = 6,
        bgcolor = "rgba(255,255,255,0.85)",
        font = PlotlyJS.attr(family = "monospace", size = 10, color = "#222"))

    # Show/hide toggle for the reference box (two-button pair, top-right).
    info_toggle = PlotlyJS.attr(type = "buttons", direction = "left", showactive = true,
        x = 1.0, y = 1.08, xanchor = "right", yanchor = "top",
        pad = PlotlyJS.attr(t = 4, l = 8), active = 0,
        buttons = [
            PlotlyJS.attr(label = "ⓘ Info", method = "relayout",
                args = [Dict{String,Any}("annotations[0].visible" => true)]),
            PlotlyJS.attr(label = "Hide", method = "relayout",
                args = [Dict{String,Any}("annotations[0].visible" => false)])])

    cmin, cmax = prop_range[color_by]
    layout = PlotlyJS.Layout(height = height,
        title = PlotlyJS.attr(text = titlefor(episodes[1]), x = 0.5, xanchor = "center",
                              y = 0.98, yanchor = "top"),
        margin = PlotlyJS.attr(t = 110, b = 240, l = 10, r = 10),
        scene = PlotlyJS.attr(aspectmode = "data"),
        coloraxis = PlotlyJS.attr(colorscale = "Viridis", cmin = cmin, cmax = cmax,
            colorbar = PlotlyJS.attr(title = PlotlyJS.attr(text = String(color_by),
                                                           side = "right"))),
        legend = PlotlyJS.attr(x = 1.0, y = 1.0, xanchor = "right", yanchor = "top"),
        annotations = [config_box],
        updatemenus = [colour_menu, playmenu, info_toggle],
        sliders = [episode_slider, water_slider])

    fig = PlotlyJS.plot(traces, layout, frames)
    file !== nothing && PlotlyJS.savefig(fig, file; format = "html")
    fig
end

end # module
