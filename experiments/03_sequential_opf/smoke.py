"""Short, meaningful integration check; not a substitute for the full study."""

import opf_bootstrap
import json
import numpy as np
from optimization import (
    Evaluator,
    baseline,
    restore_feasibility,
    optimize,
    reference_circuit,
)


def main():
    # Every operation uses identical demand and must-take transfer obligations.
    ev = Evaluator(1.0)
    a = restore_feasibility(ev, baseline(1.0, kv=345.0))
    assert a["accepted"]
    b = optimize(ev, a["result"]["x"], free_frequency=False)
    c = optimize(ev, b["result"]["x"], free_frequency=True)
    assert b["accepted"] and c["accepted"]
    assert c["result"]["loss_mw"] <= b["result"]["loss_mw"] + 1e-5
    assert b["result"]["loss_mw"] <= a["result"]["loss_mw"] + 1e-5
    checks = {}
    for name, record in [("A", a), ("B", b), ("C", c)]:
        r = record["result"]
        ref = reference_circuit(ev, r["x"])
        assert ref["verified"] and r["feasible"]
        assert abs(r["power_balance_error_mw"]) < 1e-5
        # Run the Python core at the same optimized controls: same physical state.
        py = Evaluator(1.0, engine="python").evaluate(r["x"])
        assert max(np.max(abs(r[k] - py[k])) for k in ("vh", "vl", "vd", "ss")) < 1e-7
        checks[name] = dict(
            loss_mw=r["loss_mw"],
            frequency_hz=float(r["x"][11] * 10),
            feasible=r["feasible"],
            conservation_error_mw=r["power_balance_error_mw"],
            independent_verification=ref,
        )
    out = opf_bootstrap.HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "smoke_validation.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
