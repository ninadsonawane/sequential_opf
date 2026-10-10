"""Audit numerical exports against independent bookkeeping, not the optimizer."""

from pathlib import Path
import csv
import json
import math
import hashlib
from html.parser import HTMLParser

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def load_csv(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, val in attrs:
            if (
                key in ("src", "href")
                and val
                and not val.startswith(("http:", "https:", "#"))
            ):
                self.targets.append(val)


def main():
    hourly = load_csv("hourly_results.csv")
    planning = load_csv("table1_planning.csv")
    generators = load_csv("generator_dispatch_all_hours.csv")
    converters = load_csv("converter_dispatch_all_hours.csv")
    cons = load_csv("constraint_audit_all_hours.csv")
    records = load_json("optimizer_records.json")
    assert len(hourly) == 72 and len(planning) == 10
    assert len(generators) == 576 and len(converters) == 504
    assert len({(r["case"], r["hour"]) for r in hourly}) == 72
    assert all(r["feasible"] == "True" for r in hourly + planning)
    assert all(
        r["satisfied"] == "True" and float(r["normalized_margin"]) >= -1e-6
        for r in cons
    )
    assert all(r["optimizer"]["accepted"] for r in records)
    assert all(r["optimizer"]["solver_success"] for r in records if r["case"] != "A")
    generation_errors = []
    loss_errors = []
    converter_errors = []
    for r in hourly:
        pick = lambda row: row["case"] == r["case"] and row["hour"] == r["hour"]
        pg = sum(float(g["p_generation_mw"]) for g in generators if pick(g))
        loss = float(r["total_loss_mw"])
        demand = float(r["demand_mw"])
        generation_errors.append(abs(pg - demand - loss))
        converter_errors.append(
            abs(
                sum(float(c["station_loss_mw"]) for c in converters if pick(c))
                - float(r["converter_loss_mw"])
            )
        )
        loss_errors.append(
            abs(
                sum(
                    float(r[k])
                    for k in (
                        "hvac_line_loss_mw",
                        "lfac_line_loss_mw",
                        "dc_line_loss_mw",
                        "converter_loss_mw",
                    )
                )
                - loss
            )
        )
    assert (
        max(generation_errors) < 1e-5
        and max(loss_errors) < 1e-9
        and max(converter_errors) < 1e-9
    )
    summary = load_json("summary.json")
    for case in "ABC":
        total = sum(float(r["total_loss_mw"]) for r in hourly if r["case"] == case)
        assert abs(total - summary["daily_loss_energy_mwh"][case]) < 1e-8
    for hour in range(24):
        row = {r["case"]: r for r in hourly if int(r["hour"]) == hour}
        assert (
            float(row["C"]["total_loss_mw"]) <= float(row["B"]["total_loss_mw"]) + 1e-5
        )
        assert (
            float(row["B"]["total_loss_mw"]) <= float(row["A"]["total_loss_mw"]) + 1e-5
        )
    samples = load_csv("benchmark_samples.csv")
    assert len(samples) == 40 and all(
        r["correctness_passed"] == "True" for r in samples
    )
    assert all(
        math.isfinite(float(r["seconds"])) and float(r["seconds"]) > 0 for r in samples
    )
    for r in samples:
        assert float(r["max_reference_error"]) < (
            1e-3 if r["scope"].startswith("complete") else 1e-7
        )
    checks = load_json("independent_validation.json")
    plans = load_json("planning_details.json")
    assert all(r["verified"] for r in checks)
    assert all(
        r["verification"]["verified"] and r["optimizer"]["accepted"] for r in plans
    )
    parser = References()
    parser.feed((OUT / "report.html").read_text(encoding="utf-8"))
    missing = [p for p in parser.targets if not (OUT / p).is_file()]
    assert not missing, missing
    hashes = load_json("source_hashes.json")
    assert all(
        hashlib.sha256((HERE.parent / p).read_bytes()).hexdigest() == digest
        for p, digest in hashes.items()
    )
    result = dict(
        passed=True,
        operating_points=len(hourly),
        planning_points=len(planning),
        constraint_records=len(cons),
        generator_export_conservation_error_mw=max(generation_errors),
        loss_component_error_mw=max(loss_errors),
        station_loss_export_error_mw=max(converter_errors),
        independent_operating_checks=len(checks),
        independent_planning_checks=len(plans),
        benchmark_batches=len(samples),
        local_report_links=len(parser.targets),
        missing_links=missing,
        source_hashes_match=True,
    )
    (OUT / "artifact_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
