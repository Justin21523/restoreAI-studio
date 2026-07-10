import { describe, expect, it } from "vitest";
import { allDemoJobs, demoBatch, demoEvidence, demoScenarios } from "./demo";

describe("portfolio evidence", () => {
  it("covers every production model family with verified artifacts", () => {
    expect(demoScenarios).toHaveLength(4);
    expect(demoEvidence.models).toHaveLength(7);
    expect(demoEvidence.models.every((model) => model.valid)).toBe(true);
    expect(demoScenarios.flatMap((scenario) => scenario.model_ids)).toEqual(
      expect.arrayContaining([
        "realesrgan-x4plus",
        "gfpgan-v1.4",
        "codeformer",
        "rife-v4.25",
      ]),
    );
    expect(
      demoScenarios.every(
        (scenario) =>
          scenario.artifact.input_sha256.length === 64 &&
          Object.values(scenario.artifact.outputs).every(
            (output) => output.sha256.length === 64,
          ),
      ),
    ).toBe(true);
  });

  it("includes successful and failed durable workflow states", () => {
    expect(allDemoJobs.some((job) => job.status === "succeeded")).toBe(true);
    expect(allDemoJobs.some((job) => job.error_code === "INVALID_IMAGE")).toBe(
      true,
    );
    expect(demoBatch.status).toBe("partial_failure");
    expect(demoBatch.jobs).toHaveLength(3);
  });
});
