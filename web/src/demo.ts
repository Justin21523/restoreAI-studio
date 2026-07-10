import rawScenarios from "./demo-scenarios.json";
import rawEvidence from "./demo-evidence.json";
import type {
  Artifact,
  Batch,
  DemoEvidence,
  DemoScenario,
  Job,
  JobEvent,
} from "./types";

const base = `${import.meta.env.BASE_URL}demo/scenarios`;

export const demoScenarios = (rawScenarios as unknown as DemoScenario[]).map(
  (scenario) => ({
    ...scenario,
    input: `${base}/${scenario.input}`,
    outputs: Object.fromEntries(
      Object.entries(scenario.outputs).map(([key, path]) => [
        key,
        `${base}/${path}`,
      ]),
    ),
  }),
);

export const demoEvidence = rawEvidence as DemoEvidence;

function scenarioEvents(scenario: DemoScenario): JobEvent[] {
  const started = new Date(demoEvidence.generated_at).getTime();
  let elapsed = 0;
  return scenario.events.map((event) => {
    elapsed += event.duration_ms;
    return {
      id: event.id,
      event_type: event.stage === "complete" ? "completed" : "progress",
      stage: event.stage,
      progress: event.progress,
      message: event.message,
      payload: { duration_ms: event.duration_ms, replay: true },
      created_at: new Date(started + elapsed).toISOString(),
    };
  });
}

function scenarioArtifact(scenario: DemoScenario): Artifact {
  const output = scenario.artifact.outputs[scenario.default_output];
  return {
    id: `artifact-${scenario.id}`,
    job_id: scenario.job_id,
    input_sha256: scenario.artifact.input_sha256,
    output_sha256: output.sha256,
    mime_type: scenario.kind === "image" ? "image/png" : "video/mp4",
    size_bytes: output.size_bytes,
    metadata_json: { ...scenario.metrics, replay: true },
    model_snapshot: Object.fromEntries(
      scenario.model_ids.map((id) => {
        const model = demoEvidence.models.find((item) => item.model_id === id);
        return [id, model ?? { model_id: id }];
      }),
    ),
    created_at: demoEvidence.generated_at,
    expires_at: "2099-12-31T23:59:59Z",
    deleted_at: null,
  };
}

export function scenarioJob(scenario: DemoScenario): Job {
  const artifact = scenarioArtifact(scenario);
  const elapsed = Number(scenario.metrics.elapsed_ms ?? 0);
  return {
    id: scenario.job_id,
    batch_id: null,
    retry_of_job_id: null,
    artifact_id: artifact.id,
    kind: scenario.kind,
    operation: scenario.operation,
    status: "succeeded",
    stage: "complete",
    progress: 1,
    parameters: scenario.parameters,
    original_filename:
      {
        "face-lab": "archive-portrait.jpg",
        "product-detail": "vintage-camera.jpg",
        "city-motion": "city-motion-24fps.mp4",
        "combined-video": "city-combined-12fps.mp4",
      }[scenario.id] ??
      scenario.input.split("/").at(-1) ??
      scenario.id,
    error_code: null,
    error_message: null,
    queued_at: demoEvidence.generated_at,
    started_at: demoEvidence.generated_at,
    finished_at: new Date(
      new Date(demoEvidence.generated_at).getTime() + elapsed,
    ).toISOString(),
    created_at: demoEvidence.generated_at,
    updated_at: demoEvidence.generated_at,
    artifact,
    events: scenarioEvents(scenario),
  };
}

export const demoJobs = demoScenarios.map(scenarioJob);

export const failedDemoJob: Job = {
  id: "demo-invalid-image",
  batch_id: "demo-batch-partial",
  retry_of_job_id: null,
  artifact_id: null,
  kind: "image",
  operation: "face_restore_upscale",
  status: "failed",
  stage: "failed",
  progress: 0.03,
  parameters: { face_method: "codeformer", fidelity: 0.7, scale: 2 },
  original_filename: "corrupted-archive.jpg",
  error_code: "INVALID_IMAGE",
  error_message: "invalid or unsupported image",
  queued_at: demoEvidence.generated_at,
  started_at: demoEvidence.generated_at,
  finished_at: demoEvidence.generated_at,
  created_at: demoEvidence.generated_at,
  updated_at: demoEvidence.generated_at,
  artifact: null,
  events: [
    {
      id: 1,
      event_type: "progress",
      stage: "validate",
      progress: 0.03,
      message: "Reading image header and dimensions",
      payload: { replay: true },
      created_at: demoEvidence.generated_at,
    },
    {
      id: 2,
      event_type: "failed",
      stage: "failed",
      progress: 0.03,
      message: "INVALID_IMAGE · invalid or unsupported image",
      payload: { error_code: "INVALID_IMAGE", replay: true },
      created_at: demoEvidence.generated_at,
    },
  ],
};

export const allDemoJobs = [...demoJobs, failedDemoJob];

export function demoScenarioForJob(jobId: string) {
  return demoScenarios.find((scenario) => scenario.job_id === jobId);
}

const batchJobs = [
  { ...demoJobs[0], batch_id: "demo-batch-partial" },
  { ...demoJobs[1], batch_id: "demo-batch-partial" },
  failedDemoJob,
];

export const demoBatch: Batch = {
  id: "demo-batch-partial",
  retry_of_batch_id: null,
  kind: "image",
  operation: "face_restore_upscale",
  status: "partial_failure",
  parameters: { face_method: "codeformer", fidelity: 0.7, scale: 2 },
  total_items: 3,
  succeeded_items: 2,
  failed_items: 1,
  created_at: demoEvidence.generated_at,
  updated_at: demoEvidence.generated_at,
  jobs: batchJobs,
};

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
