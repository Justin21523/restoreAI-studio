import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 30_000,
  fullyParallel: false,
  use: { trace: "retain-on-failure", ...devices["Desktop Chrome"] },
  projects: [
    { name: "demo", use: { baseURL: "http://127.0.0.1:4173" } },
    { name: "real", use: { baseURL: "http://127.0.0.1:4174" } },
  ],
  webServer: [
    {
      command: "VITE_APP_MODE=demo npm run dev -- --port 4173",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: true,
    },
    {
      command: "VITE_APP_MODE=real npm run dev -- --port 4174",
      url: "http://127.0.0.1:4174",
      reuseExistingServer: true,
    },
  ],
});
