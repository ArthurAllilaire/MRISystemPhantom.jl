"""
    SphereDescriptor

Geometry + material description of one contrast sphere. All spatial quantities
are in SI (metres). Used as the intermediate representation between the
material tables and the voxelised `KomaMRI.Phantom`.

| Field | Unit | Description |
|-------|------|-------------|
| `centre` | m | (x, y, z) sphere centre in phantom frame |
| `radius` | m | sphere radius |
| `ρ` | — | proton density (1.0 = bulk water) |
| `T1` | s | longitudinal relaxation time |
| `T2` | s | transverse relaxation time |
| `T2s` | s | T2* (effective transverse, includes B0 inhomogeneity) |
| `delta_w` | rad/s | off-resonance frequency |
| `label` | — | unique identifier, e.g. `:T1_3`, `:fid_1` |

See [`with_sphere_relaxation`](@ref), [`transform_descriptor`](@ref).
"""
struct SphereDescriptor
    centre::NTuple{3,Float64}   # m
    radius::Float64             # m
    ρ::Float64
    T1::Float64                 # s
    T2::Float64                 # s
    T2s::Float64                # s
    delta_w::Float64            # rad/s
    label::Symbol
end
