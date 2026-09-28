import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.mjs";

const outputNames = [
  "combined_tendon_angle_degrees.nii.gz",
  "combined_magic_angle_parameter.nii.gz",
  "combined_magic_angle_deviation_degrees.nii.gz",
  "combined_tendon_labels.nii.gz",
  "combined_tendon_centerlines.nii.gz",
  "tendon_angle_summary.csv"
];

let runtimePromise;

function report(message) {
  self.postMessage({ type: "progress", message });
}

async function initializeRuntime() {
  report("Loading the private browser analysis engine...");
  const pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "scipy", "scikit-image", "micropip"]);
  report("Loading NIfTI support...");
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("nibabel==5.3.2")
`);
  const response = await fetch(new URL("./tendon_geometry.py", import.meta.url));
  if (!response.ok) {
    throw new Error("The analysis module could not be loaded.");
  }
  pyodide.FS.writeFile("/home/pyodide/tendon_geometry.py", await response.text());
  return pyodide;
}

function runtime() {
  runtimePromise ??= initializeRuntime();
  return runtimePromise;
}

async function analyze({ reference, segmentations, b0 }) {
  const pyodide = await runtime();
  const jobId = crypto.randomUUID();
  const jobRoot = `/tmp/${jobId}`;
  const inputRoot = `${jobRoot}/input`;
  const outputRoot = `${jobRoot}/output`;
  pyodide.FS.mkdirTree(inputRoot);
  pyodide.FS.mkdirTree(outputRoot);

  try {
    const referencePath = `${inputRoot}/reference${reference.name.toLowerCase().endsWith(".gz") ? ".nii.gz" : ".nii"}`;
    pyodide.FS.writeFile(referencePath, new Uint8Array(reference.buffer));
    const segmentationPaths = segmentations.map((segmentation, index) => {
      const extension = segmentation.name.toLowerCase().endsWith(".gz") ? ".nii.gz" : ".nii";
      const path = `${inputRoot}/segmentation_${String(index + 1).padStart(2, "0")}${extension}`;
      pyodide.FS.writeFile(path, new Uint8Array(segmentation.buffer));
      return path;
    });

    report("Extracting centerlines and magic-angle maps...");
    pyodide.globals.set("analysis_input_json", JSON.stringify({
      referencePath,
      segmentationPaths,
      segmentationNames: segmentations.map(segmentation => segmentation.name),
      outputRoot,
      b0
    }));
    const summaryJson = await pyodide.runPythonAsync(`
import json
from pathlib import Path
import numpy as np
from tendon_geometry import analyze_segmentations

analysis_input = json.loads(analysis_input_json)
analysis_summary = analyze_segmentations(
    [Path(path) for path in analysis_input["segmentationPaths"]],
    Path(analysis_input["outputRoot"]),
    np.asarray(analysis_input["b0"], dtype=float),
    b0_basis="image",
    reference_path=Path(analysis_input["referencePath"]),
    segmentation_names=analysis_input["segmentationNames"],
)
json.dumps(analysis_summary)
`);

    const files = outputNames.map(name => ({
      name,
      buffer: pyodide.FS.readFile(`${outputRoot}/${name}`).slice().buffer
    }));
    self.postMessage(
      { type: "complete", summary: JSON.parse(summaryJson), files },
      files.map(file => file.buffer)
    );
  } finally {
    pyodide.globals.delete("analysis_input_json");
    pyodide.runPython(`
import shutil
shutil.rmtree(${JSON.stringify(jobRoot)}, ignore_errors=True)
`);
  }
}

self.onmessage = async event => {
  if (event.data?.type !== "analyze") return;
  try {
    await analyze(event.data);
  } catch (error) {
    self.postMessage({ type: "error", message: error.message || String(error) });
  }
};