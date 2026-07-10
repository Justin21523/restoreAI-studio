import { expect, test } from "@playwright/test";

const job = {
  id: "11111111-1111-1111-1111-111111111111",
  batch_id: null,
  retry_of_job_id: null,
  artifact_id: null,
  kind: "image",
  operation: "upscale",
  status: "running",
  stage: "upscale",
  progress: 0.64,
  parameters: { scale: 2 },
  original_filename: "portrait.png",
  error_code: null,
  error_message: null,
  queued_at: new Date().toISOString(),
  started_at: new Date().toISOString(),
  finished_at: null,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

test("real mode renders API-backed job history", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "real");
  await page.route("**/api/v1/jobs", (route) => route.fulfill({ json: [job] }));
  await page.goto("/#/workspace");
  await page.getByLabel("Preset").selectOption("old-photo");
  await expect(page.getByLabel("Operation")).toHaveValue(
    "face_restore_upscale",
  );
  await expect(page.getByLabel("CodeFormer fidelity")).toHaveValue("0.7");
  await page.getByLabel("Face model").selectOption("gfpgan");
  await expect(page.getByLabel("GFPGAN strength")).toHaveValue("0.8");
  await page.goto("/#/jobs");
  await expect(page.getByText("portrait.png")).toBeVisible();
  await expect(page.getByText("running", { exact: true })).toBeVisible();
  await expect(page.locator(".job-progress i")).toHaveAttribute("style", /64%/);
});
