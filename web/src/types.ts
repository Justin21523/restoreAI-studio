export type JobStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "succeeded"
  | "failed"
  | "expired";

export interface JobEvent {
  id: number;
  event_type: string;
  type?: string;
  stage: string;
  progress: number;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
  timestamp?: string;
}

export interface Artifact {
  id: string;
  job_id: string;
  input_sha256: string;
  output_sha256: string;
  mime_type: string;
  size_bytes: number;
  metadata_json: Record<string, unknown>;
  model_snapshot: Record<string, Record<string, unknown>>;
  created_at: string;
  expires_at: string;
  deleted_at: string | null;
}

export interface Job {
  id: string;
  batch_id: string | null;
  retry_of_job_id: string | null;
  artifact_id: string | null;
  kind: "image" | "video";
  operation: string;
  status: JobStatus;
  stage: string;
  progress: number;
  parameters: Record<string, unknown>;
  original_filename: string;
  error_code: string | null;
  error_message: string | null;
  queued_at: string;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  updated_at: string;
  artifact?: Artifact | null;
  events?: JobEvent[];
}

export interface Batch {
  id: string;
  retry_of_batch_id: string | null;
  kind: "image" | "video";
  operation: string;
  status: string;
  parameters: Record<string, unknown>;
  total_items: number;
  succeeded_items: number;
  failed_items: number;
  created_at: string;
  updated_at: string;
  jobs?: Job[];
}

export interface ModelStatus {
  model_id: string;
  family: string;
  path: string;
  available: boolean;
  valid: boolean;
  size_bytes: number;
  sha256: string | null;
  loaded: boolean;
  error: string | null;
}

export interface SystemStatus {
  status: "ready" | "degraded";
  database: Record<string, unknown>;
  redis: Record<string, unknown>;
  queue: Record<string, unknown>;
  worker: Record<string, unknown>;
  gpu: Record<string, unknown> | null;
  storage: Record<string, unknown>;
  models: Record<string, unknown>;
}

export interface DemoScenario {
  id: string;
  job_id: string;
  kind: "image" | "video";
  title: { en: string; zh: string };
  description: { en: string; zh: string };
  input: string;
  outputs: Record<string, string>;
  default_output: string;
  operation: string;
  model: string;
  model_ids: string[];
  parameters: Record<string, string | number | boolean>;
  metrics: Record<string, string | number>;
  artifact: {
    input_sha256: string;
    outputs: Record<
      string,
      { path: string; sha256: string; size_bytes: number }
    >;
  };
  events: Array<{
    id: number;
    stage: string;
    progress: number;
    message: string;
    duration_ms: number;
  }>;
}

export interface DemoEvidence {
  generated_at: string;
  environment: {
    gpu: string;
    driver: string;
    vram_mib: number;
    compute_capability: string;
    cuda: string;
    pytorch: string;
    fp16: boolean;
    python: string;
    platform: string;
  };
  models: Array<{
    model_id: string;
    family: string;
    path: string;
    sha256: string;
    size_bytes: number;
    valid: boolean;
  }>;
  benchmarks: {
    method: string;
    runs: Array<{
      id: string;
      cold_ms: number;
      warm_median_ms: number | null;
      peak_vram_mib: number;
    }>;
  };
  verification: {
    model_root: string;
    valid_models: number;
    invalid_models: number;
    method: string;
  };
}
