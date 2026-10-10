"""Build a portable report from new validation/timing outputs, without PDFs."""

import p3_bootstrap as bootstrap
import json
import csv
import html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = bootstrap.HERE / "results"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def tab(rows):
    return (
        "<table>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row)
            + "</tr>"
            for row in rows
        )
        + "</table>"
    )


def main():
    checks = read("verification.json")
    power = read("power_balance.json")
    with (OUT / "paper_bus_comparison.csv").open(encoding="utf-8") as f:
        bus = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    for ax, field, target in [
        (axes[0], "computed_vm_pu", "reported_vm_pu"),
        (axes[1], "computed_angle_deg", "reported_angle_deg"),
    ]:
        valid = [r for r in bus if r[field] != ""]
        xx = [int(r["bus"]) for r in valid]
        ax.plot(
            xx,
            [float(r[field]) for r in valid],
            "o-",
            label="Our disclosed reconstruction",
        )
        ax.plot(
            xx,
            [float(r[target]) for r in valid],
            "x--",
            label="P3 rounded numerical targets",
        )
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set(
        ylabel="Magnitude (p.u.)",
        title="Complete selected circuit; exact paper inputs remain unresolved",
    )
    axes[1].set(ylabel="Angle (degrees)", xlabel="Paper bus number")
    fig.savefig(OUT / "paper_comparison.png", dpi=160)
    plt.close(fig)
    timing = ""
    if (OUT / "timing_summary.csv").exists():
        with (OUT / "timing_summary.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        timing = "<h2>Warm timings on this machine</h2>" + tab(
            [["Workload", "Method", "Median ms", "Repeats"]]
            + [
                [r["scope"], r["method"], f"{float(r['median_ms']):.3f}", r["repeats"]]
                for r in rows
            ]
        )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Stage 2: reconstructed 25-bus sequential PF</title>
    <style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#22374a}}table{{border-collapse:collapse;width:100%;font-size:13px}}td{{padding:9px;border-bottom:1px solid #dce4ec}}img{{width:100%}}.scope{{padding:18px;background:#edf4f8}}a{{color:#147c91}}</style>
    <h1>Stage 2: sequential PF on the full reconstructed 25-bus circuit</h1>
    <p class="scope">Numerically verified selected model; not an exact reproduction of the authors' modified case. No author source code or source PDF is required. Downloaded third-party source programs are not redistributed.</p>
    <p>Inputs: <a href="../data/reconstructed_25bus.json">complete numerical JSON</a>. Read <a href="../README.md">method, provenance and capability limits</a>. Paper: <a href="https://doi.org/10.1016/j.ijepes.2026.112025">Kim, Nguyen &amp; Santoso (P3)</a>.</p>
    <h2>Independent expanded-circuit validation</h2>{tab([[k,f'{v:.3e}' if isinstance(v,float) else v] for k,v in checks['python'].items()])}
    <p>Separate flat start and 75-real-variable simultaneous circuit solve. This checks the selected equations, not whether our assumed inputs equal the authors' inputs.</p>
    <h2>Power accounting (MW)</h2>{tab([[k,f'{v:.6f}'] for k,v in power.items()])}
    <h2>Reported numerical targets versus reconstruction</h2><img src="paper_comparison.png" alt="Computed and reported bus voltages and angles">
    <p>LFAC results agree closely; some HVAC/DC results differ. Published outputs are comparison targets, not calibration inputs. See <a href="input_audit.json">missing-data audit</a> and all paper comparison CSVs.</p>
    <h2>Capability is a separate check</h2><p>See <a href="converter_limits.csv">converter flags</a>. This fixed-schedule stage flags violations without switching control modes. Convergence does not establish equipment feasibility.</p>
    {timing}<p>Only the initial independent HVAC solves are pool-parallel. Once tied, the 19-bus HVAC solve is coupled; LFAC, VSC and DC coordination remain serial. Same GS–Anderson source; no author-code speedup claim. Startup/JIT are excluded from warm timing.</p>
    <h2>Artifacts</h2><p><a href="verification.json">Verification</a> · <a href="solution.npz">Solution arrays</a> · <a href="outer_iterations.csv">Coordination trace</a> · <a href="power_balance.json">Conservation</a></p></html>"""
    (OUT / "report.html").write_text(body, encoding="utf-8")
    print("Created", OUT / "report.html")


if __name__ == "__main__":
    main()
