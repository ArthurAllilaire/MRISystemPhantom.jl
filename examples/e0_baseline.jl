# E0 — conventional-sequence baseline for the QalibreMD digital twin.
# PLAN.md §4 E0: run IR on every T1-array sphere, multi-TE SE on every
# T2-array sphere, fit monoexponentials, report MAPE against the manual
# values. Target: MAPE < 3 % (simulator sanity check + RL yardstick).
#
# Single-spin phantoms (0-D, no spatial encoding) are used because E0 is
# a non-spatial measurement — orders of magnitude faster than a
# voxelised sphere, matching the `01-FID.jl` pattern.

using MRISystemPhantom
using Printf

function print_table(header, rows)
    @printf "\n%s\n" header
    @printf "%-6s  %12s  %12s  %8s\n" "Sphere" "True [ms]" "Est [ms]" "|err|%"
    println("-"^44)
    for r in rows
        @printf "%-6s  %12.2f  %12.2f  %8.2f\n" r[1] r[2] r[3] r[4]
    end
end

for field in (:T3, :T15)
    @info "Running E0 at field" field
    res = run_e0(; field = field, verbose = false)

    t1_rows = [("T1-$i", 1000 * res.T1_true[i],
                           1000 * res.T1_est[i],
                           res.T1_err_pct[i]) for i in 1:14]
    t2_rows = [("T2-$i", 1000 * res.T2_true[i],
                           1000 * res.T2_est[i],
                           res.T2_err_pct[i]) for i in 1:14]

    print_table("T1 array @ $field", t1_rows)
    print_table("T2 array @ $field", t2_rows)

    @printf "\n→ %s  T1 MAPE = %.3f %%   T2 MAPE = %.3f %%\n" field res.T1_MAPE_pct res.T2_MAPE_pct
end
