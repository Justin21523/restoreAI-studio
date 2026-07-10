import { expect, test } from "@playwright/test";

test("public demo shows real precomputed scenarios and locale switching", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "demo");
  await page.goto("/");
  await expect(page.getByText("Public demo", { exact: false })).toBeVisible();
  await page.getByLabel("Demo scenario").selectOption("product-detail");
  await expect(
    page.getByRole("heading", { name: "Product detail" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Run demo flow" }).click();
  await expect(page.locator(".pipeline-progress i")).toHaveAttribute(
    "style",
    /100%/,
  );
  await page.getByRole("button", { name: "Switch language" }).click();
  await expect(page.getByText("工作區", { exact: true })).toBeVisible();
});
