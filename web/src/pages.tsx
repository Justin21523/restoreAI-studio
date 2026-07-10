import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import { api } from "./api";
import {
  Badge,
  formatBytes,
  ImageComparison,
  JobRow,
  JsonGrid,
  Timeline,
  VideoComparison,
} from "./components";
import { demoScenarios, presets } from "./demo";
import type {
  Batch,
  DemoScenario,
  Job,
  JobEvent,
  ModelStatus,
  SystemStatus,
} from "./types";

export const appMode = import.meta.env.VITE_APP_MODE ?? "demo";

function DemoPanel({
  scenario,
  progress,
}: {
  scenario: DemoScenario;
  progress: number;
}) {
  const { i18n, t } = useTranslation();
  const locale = i18n.language.startsWith("zh") ? "zh" : "en";
  return (
    <div className="demo-panel">
      <div className="demo-copy">
        <Badge status="precomputed" />
        <h2>{scenario.title[locale]}</h2>
        <p>{scenario.description[locale]}</p>
      </div>
      {scenario.kind === "image" ? (
        <ImageComparison before={scenario.input} after={scenario.output} />
      ) : (
        <VideoComparison before={scenario.input} after={scenario.output} />
      )}
      <div className="pipeline-progress">
        <i style={{ width: `${progress * 100}%` }} />
      </div>
      <p className="fine-print">{t("workspace.demoNote")}</p>
      <div className="demo-metadata">
        <JsonGrid
          value={{
            model: scenario.model,
            operation: scenario.operation,
            ...scenario.parameters,
            ...scenario.metrics,
          }}
        />
      </div>
    </div>
  );
}

export function Workspace() {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const [search] = useSearchParams();
  const queryClient = useQueryClient();
  const [kind, setKind] = useState<"image" | "video">("image");
  const [files, setFiles] = useState<File[]>([]);
  const [operation, setOperation] = useState("upscale");
  const [scale, setScale] = useState("2");
  const [faceMethod, setFaceMethod] = useState("codeformer");
  const [targetFps, setTargetFps] = useState("60");
  const [preset, setPreset] = useState("custom");
  const [scenarioId, setScenarioId] = useState(
    search.get("scenario") ?? demoScenarios[0].id,
  );
  const [demoProgress, setDemoProgress] = useState(0);
  const [message, setMessage] = useState(t("workspace.choose"));
  const scenario =
    demoScenarios.find((item) => item.id === scenarioId) ?? demoScenarios[0];
  const locale = i18n.language.startsWith("zh") ? "zh" : "en";

  useEffect(() => setMessage(t("workspace.choose")), [t]);

  const mutation = useMutation({
    mutationFn: async (form: FormData) =>
      files.length > 1 ? api.submitBatch(kind, form) : api.submit(kind, form),
    onSuccess: (result: Job | Batch) => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      if ("total_items" in result) {
        setMessage(t("workspace.batchQueued"));
        navigate(`/batches/${result.id}`);
      } else {
        setMessage(t("workspace.queued"));
        navigate(`/jobs/${result.id}`);
      }
    },
  });

  function changeKind(next: "image" | "video") {
    setKind(next);
    setOperation(next === "image" ? "upscale" : "interpolate");
    setFiles([]);
    setPreset("custom");
  }
  function selectFiles(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    setFiles(selected);
    setMessage(
      selected.length
        ? t("workspace.selected", { count: selected.length })
        : t("workspace.choose"),
    );
  }
  function applyPreset(id: string) {
    setPreset(id);
    const selected = presets.find((item) => item.id === id);
    if (!selected) return;
    setKind(selected.kind);
    setOperation(selected.operation);
    setScale(selected.scale);
    setFaceMethod(selected.faceMethod);
    setTargetFps(selected.targetFps);
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (appMode === "demo") {
      setDemoProgress(0.04);
      for (const item of scenario.events) {
        await new Promise((resolve) => window.setTimeout(resolve, 300));
        setDemoProgress(item.progress);
      }
      return;
    }
    if (!files.length) {
      setMessage(t("workspace.noFile"));
      return;
    }
    const form = new FormData();
    files.forEach((file) =>
      form.append(files.length > 1 ? "files" : "file", file),
    );
    form.append("operation", operation);
    form.append("scale", scale);
    if (kind === "image") form.append("face_method", faceMethod);
    if (kind === "video" && operation !== "upscale")
      form.append("target_fps", targetFps);
    mutation.mutate(form);
  }
  const running = mutation.isPending || (demoProgress > 0 && demoProgress < 1);
  return (
    <main className="workspace">
      <section className="hero-copy">
        <p className="eyebrow">{t("workspace.eyebrow")}</p>
        <h1>{t("workspace.title")}</h1>
        <p>{t("workspace.intro")}</p>
        <div className="mode-note">
          {t(appMode === "demo" ? "workspace.demoMode" : "workspace.realMode")}
        </div>
      </section>
      <section className="workbench panel">
        <form onSubmit={submit}>
          {appMode === "demo" ? (
            <label className="wide-control">
              {t("workspace.demoScenario")}
              <select
                value={scenarioId}
                onChange={(event) => {
                  setScenarioId(event.target.value);
                  setDemoProgress(0);
                }}
              >
                <option value="archive-portrait">
                  Archive portrait / 老照片人像
                </option>
                <option value="product-detail">
                  Product detail / 產品細節
                </option>
                <option value="city-motion">City motion / 城市動態</option>
              </select>
            </label>
          ) : (
            <>
              <div className="segmented" aria-label="Media type">
                <button
                  type="button"
                  className={kind === "image" ? "active" : ""}
                  onClick={() => changeKind("image")}
                >
                  {t("workspace.image")}
                </button>
                <button
                  type="button"
                  className={kind === "video" ? "active" : ""}
                  onClick={() => changeKind("video")}
                >
                  {t("workspace.video")}
                </button>
              </div>
              <label className="drop-zone">
                <input
                  type="file"
                  multiple
                  onChange={selectFiles}
                  accept={kind === "image" ? "image/*" : "video/*"}
                />
                <strong>
                  {files.length
                    ? t("workspace.selected", { count: files.length })
                    : t("workspace.drop")}
                </strong>
                <span>
                  {t(
                    kind === "image"
                      ? "workspace.imageLimits"
                      : "workspace.videoLimits",
                  )}
                </span>
              </label>
              <label className="wide-control">
                {t("workspace.preset")}
                <select
                  value={preset}
                  onChange={(event) => applyPreset(event.target.value)}
                >
                  <option value="custom">{t("workspace.custom")}</option>
                  {presets.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label[locale]}
                    </option>
                  ))}
                </select>
              </label>
              <div className="controls">
                <label>
                  {t("workspace.operation")}
                  <select
                    value={operation}
                    onChange={(event) => {
                      setOperation(event.target.value);
                      setPreset("custom");
                    }}
                  >
                    {kind === "image" ? (
                      <>
                        <option value="upscale">
                          {t("workspace.superResolution")}
                        </option>
                        <option value="face_restore">
                          {t("workspace.faceRestore")}
                        </option>
                        <option value="face_restore_upscale">
                          {t("workspace.faceUpscale")}
                        </option>
                      </>
                    ) : (
                      <>
                        <option value="interpolate">
                          {t("workspace.interpolate")}
                        </option>
                        <option value="upscale">
                          {t("workspace.videoUpscale")}
                        </option>
                        <option value="interpolate_upscale">
                          {t("workspace.combined")}
                        </option>
                      </>
                    )}
                  </select>
                </label>
                <label>
                  {t("workspace.scale")}
                  <select
                    value={scale}
                    onChange={(event) => setScale(event.target.value)}
                  >
                    <option value="2">2×</option>
                    <option value="4">4×</option>
                  </select>
                </label>
                {kind === "image" && operation !== "upscale" ? (
                  <label>
                    {t("workspace.faceModel")}
                    <select
                      value={faceMethod}
                      onChange={(event) => setFaceMethod(event.target.value)}
                    >
                      <option value="codeformer">CodeFormer</option>
                      <option value="gfpgan">GFPGAN</option>
                    </select>
                  </label>
                ) : null}
                {kind === "video" && operation !== "upscale" ? (
                  <label>
                    {t("workspace.targetFps")}
                    <input
                      type="number"
                      min="1"
                      max="120"
                      value={targetFps}
                      onChange={(event) => setTargetFps(event.target.value)}
                    />
                  </label>
                ) : null}
              </div>
            </>
          )}
          {appMode === "demo" ? (
            <DemoPanel scenario={scenario} progress={demoProgress} />
          ) : null}
          <div className="submit-row">
            <output aria-live="polite">
              {mutation.error ? mutation.error.message : message}
            </output>
            <button className="primary" disabled={running}>
              {running
                ? t("workspace.processing")
                : t(
                    appMode === "demo"
                      ? "workspace.demoRun"
                      : "workspace.queue",
                  )}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

export function History() {
  const { t, i18n } = useTranslation();
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    enabled: appMode === "real",
    refetchInterval: 1500,
  });
  const locale = i18n.language.startsWith("zh") ? "zh" : "en";
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <p className="eyebrow">{t("jobs.eyebrow")}</p>
          <h1>{t("jobs.title")}</h1>
        </div>
        {jobs.isFetching && <span>{t("jobs.refreshing")}</span>}
      </div>
      <section className="panel job-list">
        {appMode === "demo" ? (
          demoScenarios.map((item) => (
            <article className="demo-job" key={item.id}>
              <div>
                <Badge status="succeeded" />
                <h2>{item.title[locale]}</h2>
                <p>{item.model}</p>
              </div>
              <Link className="small-button" to={`/?scenario=${item.id}`}>
                Demo
              </Link>
            </article>
          ))
        ) : jobs.isError ? (
          <p className="error">{jobs.error.message}</p>
        ) : jobs.data?.length ? (
          jobs.data.map((job) => <JobRow key={job.id} job={job} />)
        ) : (
          <p className="empty">{t("jobs.empty")}</p>
        )}
      </section>
    </main>
  );
}

export function JobDetail() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { jobId = "" } = useParams();
  const queryClient = useQueryClient();
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    enabled: appMode === "real",
    refetchInterval: (query) =>
      ["succeeded", "failed", "cancelled", "expired"].includes(
        query.state.data?.status ?? "",
      )
        ? false
        : 1200,
  });
  const action = useMutation<Job | void, Error, "cancel" | "retry" | "delete">({
    mutationFn: (type) =>
      type === "cancel"
        ? api.cancel(jobId)
        : type === "retry"
          ? api.retry(jobId)
          : api.deleteArtifact(job.data?.artifact?.id ?? ""),
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: ["job", jobId] });
      if (result && result.id !== jobId) navigate(`/jobs/${result.id}`);
      if (!result) navigate("/jobs");
    },
  });
  if (appMode === "demo")
    return (
      <main className="page">
        <p className="empty">{t("workspace.realOnly")}</p>
      </main>
    );
  if (job.isLoading)
    return (
      <main className="page">
        <p>{t("common.loading")}</p>
      </main>
    );
  if (!job.data || job.isError)
    return (
      <main className="page">
        <p className="error">{job.error?.message ?? t("common.noData")}</p>
      </main>
    );
  const item = job.data;
  const artifact = item.artifact;
  const preview =
    artifact && !artifact.deleted_at && item.status === "succeeded";
  return (
    <main className="page detail-page">
      <div className="page-title">
        <div>
          <Link to="/jobs">← {t("common.back")}</Link>
          <p className="eyebrow">{t("detail.job")}</p>
          <h1>{item.original_filename}</h1>
        </div>
        <Badge status={item.status} />
      </div>
      {preview ? (
        item.kind === "image" ? (
          <ImageComparison
            before={api.input(item.id)}
            after={api.download(artifact.id)}
          />
        ) : (
          <VideoComparison
            before={api.input(item.id)}
            after={api.download(artifact.id)}
          />
        )
      ) : (
        <section className="panel empty">{t("detail.noPreview")}</section>
      )}
      <div className="detail-actions">
        {["queued", "running", "cancelling"].includes(item.status) ? (
          <button onClick={() => action.mutate("cancel")}>
            {t("common.cancel")}
          </button>
        ) : null}
        {["failed", "cancelled"].includes(item.status) ? (
          <button onClick={() => action.mutate("retry")}>
            {t("common.retry")}
          </button>
        ) : null}
        {artifact ? (
          <>
            <a className="primary-link" href={api.download(artifact.id)}>
              {t("common.download")}
            </a>
            <button className="danger" onClick={() => action.mutate("delete")}>
              {t("common.delete")}
            </button>
          </>
        ) : null}
      </div>
      <div className="detail-grid">
        <section className="panel detail-card">
          <h2>{t("detail.parameters")}</h2>
          <JsonGrid
            value={{
              operation: item.operation,
              ...item.parameters,
              progress: `${Math.round(item.progress * 100)}%`,
              error_code: item.error_code ?? "—",
            }}
          />
        </section>
        {artifact ? (
          <section className="panel detail-card">
            <h2>{t("detail.provenance")}</h2>
            <JsonGrid
              value={{
                size: formatBytes(artifact.size_bytes),
                input_sha256: artifact.input_sha256,
                output_sha256: artifact.output_sha256,
                expires_at: new Date(artifact.expires_at).toLocaleString(),
                models: Object.keys(artifact.model_snapshot).join(", "),
                ...artifact.metadata_json,
              }}
            />
          </section>
        ) : null}
      </div>
      <section className="panel detail-card">
        <h2>{t("detail.timeline")}</h2>
        <Timeline events={item.events ?? []} liveUrl={api.events(item.id)} />
      </section>
    </main>
  );
}

export function BatchDetail() {
  const { t } = useTranslation();
  const { batchId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const batch = useQuery({
    queryKey: ["batch", batchId],
    queryFn: () => api.batch(batchId),
    enabled: appMode === "real",
    refetchInterval: 1500,
  });
  const action = useMutation({
    mutationFn: (type: "cancel" | "retry") =>
      type === "cancel" ? api.cancelBatch(batchId) : api.retryBatch(batchId),
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: ["batch", batchId] });
      if (result.id !== batchId) navigate(`/batches/${result.id}`);
    },
  });
  if (appMode === "demo")
    return (
      <main className="page">
        <p className="empty">{t("workspace.realOnly")}</p>
      </main>
    );
  if (!batch.data)
    return (
      <main className="page">
        <p className={batch.isError ? "error" : "empty"}>
          {batch.error?.message ?? t("common.loading")}
        </p>
      </main>
    );
  const item = batch.data;
  const progress = item.total_items
    ? (item.succeeded_items + item.failed_items) / item.total_items
    : 0;
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <Link to="/jobs">← {t("common.back")}</Link>
          <p className="eyebrow">{t("detail.batch")}</p>
          <h1>{item.id.slice(0, 8)}</h1>
        </div>
        <Badge status={item.status} />
      </div>
      <section className="panel batch-summary">
        <h2>{t("detail.batchProgress")}</h2>
        <div className="pipeline-progress">
          <i style={{ width: `${progress * 100}%` }} />
        </div>
        <JsonGrid
          value={{
            total: item.total_items,
            succeeded: item.succeeded_items,
            failed: item.failed_items,
            operation: item.operation,
            retry_of: item.retry_of_batch_id ?? "—",
          }}
        />
        <div className="detail-actions">
          {["queued", "running", "cancelling"].includes(item.status) ? (
            <button onClick={() => action.mutate("cancel")}>
              {t("common.cancel")}
            </button>
          ) : null}
          {item.failed_items > 0 ? (
            <button onClick={() => action.mutate("retry")}>
              {t("detail.failedOnly")}
            </button>
          ) : null}
        </div>
      </section>
      <section className="panel job-list">
        {item.jobs?.map((job) => (
          <JobRow key={job.id} job={job} />
        ))}
      </section>
    </main>
  );
}

export function Models() {
  const { t } = useTranslation();
  const models = useQuery({
    queryKey: ["models"],
    queryFn: api.models,
    enabled: appMode === "real",
  });
  const demoModels: ModelStatus[] = [
    "realesrgan-x4plus",
    "gfpgan-v1.4",
    "codeformer",
    "rife-v4.25",
  ].map((model_id) => ({
    model_id,
    family: "demo",
    path: "/mnt/c/ai_models/…",
    available: true,
    valid: true,
    size_bytes: 0,
    sha256: null,
    loaded: false,
    error: null,
  }));
  const rows = appMode === "demo" ? demoModels : (models.data ?? []);
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <p className="eyebrow">{t("models.eyebrow")}</p>
          <h1>{t("models.title")}</h1>
        </div>
      </div>
      <section className="model-grid">
        {rows.map((model) => (
          <article className="panel model-card" key={model.model_id}>
            <Badge status={model.valid ? "ready" : "invalid"} />
            <h2>{model.model_id}</h2>
            <p>{model.family}</p>
            <code>{model.path}</code>
            {model.sha256 && (
              <small>sha256 · {model.sha256.slice(0, 16)}…</small>
            )}
          </article>
        ))}
      </section>
    </main>
  );
}

export function SystemPage() {
  const { t } = useTranslation();
  const status = useQuery<SystemStatus>({
    queryKey: ["system"],
    queryFn: api.system,
    enabled: appMode === "real",
    refetchInterval: 5000,
  });
  const demo: SystemStatus = {
    status: "ready",
    database: { status: "demo" },
    redis: { status: "demo" },
    queue: { queued: 0, running: 0 },
    worker: { status: "local-only" },
    gpu: { device_name: "RTX 5080 (local real mode)", cuda_available: true },
    storage: { mode: "browser-only" },
    models: { total: 7, valid: 7 },
  };
  const data = appMode === "demo" ? demo : status.data;
  if (!data)
    return (
      <main className="page">
        <p>{t("common.loading")}</p>
      </main>
    );
  const cards = [
    ["database", t("system.database")],
    ["redis", t("system.redis")],
    ["queue", t("system.queue")],
    ["worker", t("system.worker")],
    ["gpu", t("system.gpu")],
    ["storage", t("system.storage")],
  ] as const;
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <p className="eyebrow">{t("system.eyebrow")}</p>
          <h1>{t("system.title")}</h1>
        </div>
        <Badge status={data.status} />
      </div>
      <section className="status-grid">
        {cards.map(([key, label]) => (
          <article className="panel status-card" key={key}>
            <h2>{label}</h2>
            <JsonGrid value={(data[key] ?? {}) as Record<string, unknown>} />
          </article>
        ))}
      </section>
    </main>
  );
}
