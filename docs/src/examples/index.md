# Example Gallery

The package ships a set of runnable, self-contained demonstration scripts in
[`examples/`](https://github.com/ArthurAllilaire/MRISystemPhantom.jl/tree/main/examples).
Each exercises one slice of the public API; the plotting scripts write
interactive Plotly HTML into `src/assets/`. Run any of them with:

```bash
julia --project=. examples/<name>.jl
```

| Script | What it does |
|--------|--------------|
| [`conventional_baseline.jl`](conventional_baseline.md) | Runs the conventional fixed-sequence baseline — IR/SE sweep plus fit for all 28 T1 & T2 contrast spheres — and reports MAPE against the manual values. |
| [`t1_mapping.jl`](t1_mapping.md) | Runs a full IR-SE acquisition, reconstructs the image, extracts per-sphere ROIs and fits T1. |
| [`snr_calibration.jl`](snr_calibration.md) | Performs the NEMA MS-1 dual-acquisition SNR measurement. |
| [`compare_water_coarseness.jl`](compare_water_coarseness.md) | Compares image, k-space and per-sphere ROI signal across background-water grid coarseness (the fidelity-vs-cost trade-off). |
| `hamming_apodisation.jl` | Reconstructs a coarse-water phantom with and without a 2-D Hamming window, showing how apodisation suppresses truncation ringing. |
| `plot_phantom.jl` | Renders interactive full-phantom, plate-slice, vertical-slice and augmented-phantom views with `plot_phantom_html`. |
| `plot_fidelity_phantoms.jl` | Visualises the water-coarsening fidelity levels of the multi-fidelity curriculum with one interactive viewer per fidelity. |
| `plot_seqs.jl` | Writes one interactive RF/gradient/ADC waveform plot per sequence into `src/assets/sequences/`: IR-SE (plain and with a crusher & TR-spoiler variant), SE for T2 mapping, IR-TSE echo-train, and spoiled GRE. |

The linked scripts have rendered walkthroughs in the pages below; the rest are
short plotting utilities best read directly from the source. Phantom-viewer
examples use the package's PlotlyJS extension, while sequence and reconstruction
examples use standard KomaMRI/PlotlyJS plotting.
