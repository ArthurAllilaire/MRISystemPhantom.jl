# MRISystemPhantom.jl

A programmatic, parameterised digital twin of the
[Caliber MRI NIST/ISMRM System Standard Model 130](https://www.qmri.com/product/premium-system-phantom/)
MRI phantom, built on [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl).

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://arthurallilaire.github.io/MRISystemPhantom.jl/dev/)

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ArthurAllilaire/MRISystemPhantom.jl")
```

## Quick start

```julia
using MRISystemPhantom, KomaMRI

cfg     = PhantomConfig(field = :T3, voxel_size_mm = 2.0)
obj     = build_phantom(cfg)
scanner = scanner_for_field(cfg)
seq     = ir_sequence(0.4)
raw     = simulate(obj, seq, scanner)
```

## Documentation

Full documentation at https://arthurallilaire.github.io/MRISystemPhantom.jl/dev/
