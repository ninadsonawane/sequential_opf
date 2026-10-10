"""Check public artifact links and the absence of hidden workspace dependencies."""

from pathlib import Path
from html.parser import HTMLParser
import json
import re

ROOT = Path(__file__).resolve().parent.parent


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links += [v for k, v in attrs if k in ("href", "src") and v]


def main():
    paths = [
        ROOT / "README.md",
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("experiments/*/README.md"),
        *ROOT.glob("experiments/*/data/README.md"),
        *ROOT.glob("experiments/*/recorded_results/report.html"),
    ]
    missing = []
    checked = 0
    for file in paths:
        text = file.read_text(encoding="utf-8")
        if file.suffix == ".html":
            parser = Links()
            parser.feed(text)
            links = parser.links
        else:
            links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        for link in links:
            if link.startswith(("http:", "https:", "#", "mailto:")):
                continue
            checked += 1
            target = file.parent / link.split("#", 1)[0]
            if not target.exists():
                missing.append(str(file.relative_to(ROOT)) + ": " + link)
    assert not missing, missing
    for file in ROOT.glob("experiments/*/*.py"):
        code = file.read_text(encoding="utf-8")
        assert not re.search(r"C:[/\\]", code), f"Machine-specific path in {file}"
        assert 'sys.path.insert(0, str(HERE / "_deps"))' not in code
    data = json.loads(
        (
            ROOT / "experiments/02_sequential_pf_25bus/data/reconstructed_25bus.json"
        ).read_text()
    )
    assert len(data["hvac"]["bus"]) == 19 and len(data["lfac"]["bus"]) == 3
    assert len(data["dc_ybus"]) == 3 and len(data["initial_grids"]) == 2
    print(
        json.dumps(
            dict(
                passed=True,
                checked_local_links=checked,
                missing_links=missing,
                machine_specific_source_paths=False,
                bundled_bus_count=25,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
