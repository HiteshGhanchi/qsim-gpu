// Install a qsim-gpu wheel into Pyodide (under Node.js) and print the workload's results as JSON.
//   node check.mjs ../../dist/qsim_gpu-0.2.0-py3-none-any.whl pyodide_out.json [--timings]
// (The result goes to a file because Pyodide prints its own loading messages on stdout.)
// Compare with CPython:  python scripts/compare_pyodide.py web/pyodide-check/pyodide_out.json
import { loadPyodide } from "pyodide";
import { readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const [wheelPath, outPath] = process.argv.slice(2);
if (!wheelPath || !outPath) throw new Error("usage: node check.mjs <wheel> <out.json> [--timings]");
const withTimings = process.argv.includes("--timings");
const here = dirname(fileURLToPath(import.meta.url));

const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "micropip"]);
const name = basename(wheelPath);
pyodide.FS.writeFile(`/tmp/${name}`, readFileSync(wheelPath));
await pyodide.runPythonAsync(`import micropip\nawait micropip.install("emfs:/tmp/${name}")`);
pyodide.FS.writeFile("/tmp/workload.py", readFileSync(join(here, "workload.py"), "utf8"));
const out = pyodide.runPython(
  `import sys\nsys.path.insert(0, "/tmp")\nimport workload\nworkload.main_json(${withTimings ? "True" : "False"})`);
writeFileSync(outPath, out);
console.log(`wrote ${outPath}`);
