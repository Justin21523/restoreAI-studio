import { expect, test } from "@playwright/test";

test.beforeEach(({}, testInfo) => {
  test.skip(testInfo.project.name !== "demo");
});

test("showcase proves four model families and recorded GPU evidence", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      name: "Four real models. One production-shaped workflow.",
    }),
  ).toBeVisible();
  for (const model of ["Real-ESRGAN", "GFPGAN", "CodeFormer", "RIFE v4.25"]) {
    await expect(page.getByText(model, { exact: true }).first()).toBeVisible();
  }
  await expect(page.getByText("NVIDIA GeForce RTX 5080")).toBeVisible();
  await expect(page.locator(".scenario-card")).toHaveCount(4);
});

test("interactive lab replays a real result and opens Artifact detail", async ({
  page,
}) => {
  await page.goto("/#/workspace?scenario=product-detail");
  await expect(
    page.getByRole("heading", { name: "4× product detail" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Run demo flow" }).click();
  await expect(page.locator(".pipeline-progress i")).toHaveAttribute(
    "style",
    /100%/,
  );
  await page.getByRole("link", { name: "Inspect Job & Artifact" }).click();
  await expect(
    page.getByRole("heading", { name: "vintage-camera.jpg" }),
  ).toBeVisible();
  await expect(page.getByText("Artifact provenance")).toBeVisible();
  await expect(page.getByText("realesrgan-x4plus")).toBeVisible();
});

test("public workflow exposes partial failure, error and retry lineage", async ({
  page,
}) => {
  await page.goto("/#/jobs");
  await page.getByText("Batch partial failure replay").click();
  await expect(
    page.getByText("partial failure", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Retry failed" }).click();
  await expect(page.getByText("demo-batch-retry-01")).toBeVisible();
  await page.getByRole("link", { name: "Inspect" }).last().click();
  await expect(page.getByText("INVALID_IMAGE").first()).toBeVisible();
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("demo-retry-invalid-1")).toBeVisible();
});

test("locale switch keeps the responsive showcase usable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto("/");
  await page.getByRole("button", { name: "Switch language" }).click();
  await expect(page.getByText("成果展示", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "四個真實模型，一套完整工作流程。" }),
  ).toBeVisible();
});
