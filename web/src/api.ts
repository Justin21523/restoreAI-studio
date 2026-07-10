import type { Artifact, Batch, Job, ModelStatus, SystemStatus } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, init);
  if (!response.ok) {
    const body = await response.text();
    let message = body;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      message = parsed.detail ?? body;
    } catch {
      // Preserve plain-text server errors.
    }
    throw new Error(message || `${response.status} ${response.statusText}`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  jobs: () => request<Job[]>("/jobs"),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  batch: (id: string) => request<Batch>(`/batches/${id}`),
  artifact: (id: string) => request<Artifact>(`/artifacts/${id}`),
  models: () => request<ModelStatus[]>("/models"),
  system: () => request<SystemStatus>("/system/status"),
  submit: (kind: "image" | "video", form: FormData) =>
    request<Job>(`/jobs/${kind}s`, { method: "POST", body: form }),
  submitBatch: (kind: "image" | "video", form: FormData) =>
    request<Batch>(`/batches/${kind}s`, { method: "POST", body: form }),
  cancel: (id: string) =>
    request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  retry: (id: string) => request<Job>(`/jobs/${id}/retry`, { method: "POST" }),
  cancelBatch: (id: string) =>
    request<Batch>(`/batches/${id}/cancel`, { method: "POST" }),
  retryBatch: (id: string) =>
    request<Batch>(`/batches/${id}/retry-failed`, { method: "POST" }),
  deleteArtifact: (id: string) =>
    request<void>(`/artifacts/${id}`, { method: "DELETE" }),
  input: (jobId: string) => `/api/v1/jobs/${jobId}/input`,
  download: (artifactId: string) => `/api/v1/artifacts/${artifactId}/download`,
  events: (jobId: string) => `/api/v1/jobs/${jobId}/events`,
};
