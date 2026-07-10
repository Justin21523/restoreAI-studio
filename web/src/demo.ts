import rawScenarios from "./demo-scenarios.json";
import type { DemoScenario } from "./types";

const base = `${import.meta.env.BASE_URL}demo/scenarios`;

export const demoScenarios = (rawScenarios as unknown as DemoScenario[]).map(
  (scenario) => ({
    ...scenario,
    input: `${base}/${scenario.input}`,
    output: `${base}/${scenario.output}`,
  }),
);

export const presets = [
  {
    id: "old-photo",
    label: { en: "Old Photo Restore", zh: "老照片修復" },
    kind: "image",
    operation: "face_restore_upscale",
    scale: "2",
    faceMethod: "codeformer",
    targetFps: "60",
  },
  {
    id: "portrait",
    label: { en: "Portrait Repair", zh: "人像修復" },
    kind: "image",
    operation: "face_restore",
    scale: "2",
    faceMethod: "gfpgan",
    targetFps: "60",
  },
  {
    id: "anime",
    label: { en: "Anime Upscale", zh: "動漫超解析" },
    kind: "image",
    operation: "upscale",
    scale: "4",
    faceMethod: "codeformer",
    targetFps: "60",
  },
  {
    id: "product",
    label: { en: "Product Image 2×", zh: "產品圖片 2 倍" },
    kind: "image",
    operation: "upscale",
    scale: "2",
    faceMethod: "codeformer",
    targetFps: "60",
  },
  {
    id: "smooth-video",
    label: { en: "Smooth Video 60 FPS", zh: "流暢影片 60 FPS" },
    kind: "video",
    operation: "interpolate",
    scale: "2",
    faceMethod: "codeformer",
    targetFps: "60",
  },
  {
    id: "restore-video",
    label: { en: "Video Restore + Upscale", zh: "影片修復＋超解析" },
    kind: "video",
    operation: "interpolate_upscale",
    scale: "2",
    faceMethod: "codeformer",
    targetFps: "60",
  },
] as const;
