"""Generate the README result tables from stored benchmark JSON (never type numbers by hand).

    python -m bench.report                       # newest GPU folder that has results
    python -m bench.report --gpu tesla-t4        # a specific folder under bench/results/
    python -m bench.report --print               # print the tables instead of editing README

The tables replace everything between the two marker comments in README.md:
    <!-- report:start -->  ...  <!-- report:end -->
Files marked "suspect" (unhealthy GPU) are refused unless --allow-suspect is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
README = ROOT / "README.md"
START, END = "<!-- report:start -->", "<!-- report:end -->"


class Results:
    def __init__(self, folder: Path, allow_suspect: bool) -> None:
        self.folder = folder
        self.allow_suspect = allow_suspect

    def get(self, name: str) -> dict[str, Any] | None:
        path = self.folder / f"{name}.json"
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        suspect = data.get("suspect", "warning" in data["environment"].get("gpu", {}))
        if suspect and not self.allow_suspect:
            sys.exit(f"{path} is marked suspect (unhealthy GPU); refusing to report it")
        return data

    def rel(self, name: str) -> str:
        return f"`bench/results/{self.folder.name}/{name}.json`"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _ms(x: float) -> str:
    return f"{x:.3f}" if x < 1 else f"{x:.1f}" if x < 100 else f"{x:.0f}"


def section_env(r: Results) -> list[str]:
    env = (r.get("env") or r.get("b0_bandwidth") or {}).get("environment", {})
    gpu = env.get("gpu", {})
    bw = env.get("bandwidth_check", {})
    lines = [f"**Hardware:** {gpu.get('name', '?')} ({gpu.get('memory_total', '?')}), "
             f"driver {gpu.get('driver', '?')}, CUDA runtime {gpu.get('cuda_runtime', '?')}, "
             f"Python {env.get('python', '?')}, commit `{env.get('git_commit', '?')}`."]
    if bw.get("published_GBps"):
        lines.append(f"Bandwidth check: a 64 MiB copy reached {bw['copy_64MiB_GBps']:.0f} GB/s, "
                     f"{100 * bw['fraction_of_published']:.0f}% of the published "
                     f"{bw['published_GBps']:.0f} GB/s ({bw['published_source']}).")
    return lines


def section_bandwidth(r: Results) -> list[str]:
    b0, b2, b3 = r.get("b0_bandwidth"), r.get("b2_scaling_complex64"), r.get(
        "b3_efficiency_general")
    out: list[str] = []
    if b0 and b3:
        h = [row for row in b3["rows"] if row["case"].startswith("h_")]
        lo = min(100 * row["fraction_of_copy"] for row in h)
        hi = max(100 * row["fraction_of_copy"] for row in h)
        out.append(f"Copy bandwidth (B0): **{b0['achievable_GBps']:.0f} GB/s**. "
                   f"The general single-qubit kernel on {b3['n']} qubits runs at "
                   f"**{lo:.1f}–{hi:.1f}% of a copy of the same state** (B3). "
                   f"Sources: {r.rel('b0_bandwidth')}, {r.rel('b3_efficiency_general')}.")
    if b2:
        rows = [[str(x["n"]), _ms(x["ms_per_gate"]), f"{x['GBps']:.0f}"]
                for x in b2["rows"] if x["n"] in (10, 14, 18, 20, 22, 24, 26, 28, 30)]
        src = r.rel("b2_scaling_complex64")
        out += ["", f"Time for one H gate vs number of qubits (B2, {src}):",
                "", *_table(["Qubits", "ms per gate", "effective GB/s"], rows)]
    return out


def section_specialised(r: Results) -> list[str]:
    b4 = r.get("b4_specialised")
    if not b4:
        return []
    rows = [[x["case"], _ms(x["general"]["ms_per_gate"]), _ms(x["specialized"]["ms_per_gate"]),
             f"{x['speedup']:.2f}×"] for x in b4["rows"]]
    return [f"Specialised kernels vs the general kernel, {b4['n']} qubits, same run "
            f"(B4, {r.rel('b4_specialised')}):", "",
            *_table(["Gate", "General (ms)", "Specialised (ms)", "Speedup"], rows)]


def section_fusion(r: Results) -> list[str]:
    b5 = r.get("b5_fusion")
    if not b5:
        return []
    rows = [[x["circuit"], x["variant"], str(x["passes"]), _ms(x["timing"]["median_ms"]),
             f"{x['speedup_vs_unfused_general']:.2f}×"] for x in b5["rows"]]
    return [f"Gate fusion, {b5['n']} qubits (B5, {r.rel('b5_fusion')}):", "",
            *_table(["Circuit", "Variant", "Passes", "Time (ms)", "vs general"], rows)]


def section_precision(r: Results) -> list[str]:
    b6 = r.get("b6_precision")
    if not b6:
        return []
    rows = [[str(x["n"]), _ms(x["complex64_ms_per_gate"]), _ms(x["complex128_ms_per_gate"]),
             f"{x['ratio_128_over_64']:.2f}×"] for x in b6["speed_h_gate"]]
    err = max(x["complex64_max_abs_error"] for x in b6["accuracy"])
    return [f"Precision (B6, {r.rel('b6_precision')}). Largest complex64 error vs the "
            f"complex128 reference: {err:.1e}.", "",
            *_table(["Qubits", "complex64 (ms/gate)", "complex128 (ms/gate)", "Ratio"], rows)]


def section_compare(r: Results) -> list[str]:
    engines = ["aer-cpu", "aer-cpu-nofusion", "aer-gpu", "aer-gpu-nofusion", "custatevec",
               "qsim", "qsim-fused"]
    data = {e: r.get(f"b7_compare_{e}") for e in engines}
    if not data["qsim"]:
        return []
    t = {e: {(x["circuit"], x["n"]): x["timing"]["median_ms"] for x in d["rows"]}
         for e, d in data.items() if d}

    def best(*names: str) -> dict[tuple[str, int], float]:
        have = [t[n] for n in names if n in t]
        return {k: min(m[k] for m in have if k in m) for k in have[0]} if have else {}

    cols = {"Aer CPU¹": best("aer-cpu", "aer-cpu-nofusion"),
            "Aer GPU¹": best("aer-gpu", "aer-gpu-nofusion"),
            "cuStateVec²": t.get("custatevec", {}), "qsim": t["qsim"],
            "qsim + fusion": t.get("qsim-fused", {})}
    keys = sorted(t["qsim"], key=lambda k: (k[0], k[1]))
    rows = [[k[0], str(k[1])] + [_ms(c[k]) if k in c else "–" for c in cols.values()]
            for k in keys if k[1] >= 18]
    present = [e for e in engines if data[e]]
    return [f"Same circuits on other simulators, single precision, every engine returns the same "
            f"⟨Z⟩ value, full state simulated, median of 7 runs, times in ms "
            f"(B7, `bench/results/{r.folder.name}/b7_compare_*.json`; engines run: "
            f"{', '.join(present)}):", "",
            *_table(["Circuit", "Qubits", *cols], rows), "",
            "¹ Faster of the runs with and without Aer's gate fusion. "
            "² cuStateVec driven gate by gate through its general `apply_matrix` call."]


def section_capacity(r: Results) -> list[str]:
    out = []
    for dt in ("complex64", "complex128"):
        d = r.get(f"b8_scale_{dt}")
        if d:
            failed = [x["n"] for x in d["rows"] if not x.get("ok")]
            tail = f"; {failed[0]} qubits ran out of memory" if failed else ""
            out.append(f"- {dt}: **{d['max_qubits']} qubits** verified by norm and sampling{tail} "
                       f"({r.rel(f'b8_scale_{dt}')}).")
    return ["Capacity (B8):", "", *out] if out else []


def section_vqe(r: Results) -> list[str]:
    out = []
    for name, label in (("vqe_h2_4q_numpy", "NumPy, complex128"),
                        ("vqe_h2_4q_cuda", "GPU, complex64")):
        d = r.get(name)
        if d:
            out.append(f"- {label}: worst |VQE − exact| over {len(d['points'])} bond lengths "
                       f"**{d['max_abs_error']:.1e} Ha**, minimum at "
                       f"{d['minimum']['bond_length']} Å ({r.rel(name)}).")
    return ["H₂ VQE (chemical accuracy is 1.6e-3 Ha):", "", *out] if out else []


def build(r: Results) -> str:
    parts = [section_env(r), section_bandwidth(r), section_specialised(r), section_fusion(r),
             section_precision(r), section_compare(r), section_capacity(r), section_vqe(r)]
    body: list[str] = []
    for p in parts:
        if p:
            body += [*p, ""]
    return "\n".join(["_Generated by `python -m bench.report` from "
                      f"`bench/results/{r.folder.name}/`. Do not edit by hand._", "", *body])


def newest_folder() -> Path:
    # GPU folders only ("browser" holds the Pyodide measurements, reported separately)
    folders = [f for f in RESULTS.iterdir()
               if f.is_dir() and f.name != "browser" and any(f.glob("*.json"))]
    if not folders:
        sys.exit("no results found under bench/results/")
    return max(folders, key=lambda f: max(p.stat().st_mtime for p in f.glob("*.json")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", help="folder name under bench/results/")
    ap.add_argument("--print", action="store_true", dest="print_only")
    ap.add_argument("--allow-suspect", action="store_true")
    args = ap.parse_args()
    folder = RESULTS / args.gpu if args.gpu else newest_folder()
    text = build(Results(folder, args.allow_suspect))
    if args.print_only:
        print(text)
        return
    readme = README.read_text()
    if START not in readme or END not in readme:
        sys.exit(f"README.md has no {START} / {END} markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{START}\n{text}\n{END}{tail}")
    print(f"README.md updated from {folder}")


if __name__ == "__main__":
    main()
