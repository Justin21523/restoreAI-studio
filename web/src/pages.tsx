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
  ModelComparison,
  Timeline,
  VideoComparison,
} from "./components";
import {
  allDemoJobs,
  demoBatch,
  demoEvidence,
  demoJobs,
  demoScenarioForJob,
  demoScenarios,
  failedDemoJob,
  presets,
} from "./demo";
import type {
  Batch,
  DemoScenario,
  Job,
  ModelStatus,
  SystemStatus,
} from "./types";

export const appMode = import.meta.env.VITE_APP_MODE ?? "demo";

function localized(value: { en: string; zh: string }, language: string) {
  return value[language.startsWith("zh") ? "zh" : "en"];
}

export function Showcase() {
  const { t, i18n } = useTranslation();
  const buildSha = import.meta.env.VITE_BUILD_SHA ?? "local";
  const decisions = [
    [t("showcase.outbox"), t("showcase.outboxCopy")],
    [t("showcase.gpuOwner"), t("showcase.gpuOwnerCopy")],
    [t("showcase.provenance"), t("showcase.provenanceCopy")],
  ];
  return (
    <main className="showcase">
      <section className="showcase-hero">
        <div>
          <p className="eyebrow">{t("showcase.eyebrow")}</p>
          <h1>{t("showcase.title")}</h1>
          <p className="showcase-intro">{t("showcase.intro")}</p>
          <div className="hero-actions">
            <Link className="primary-link" to="/workspace">
              {t("showcase.openLab")}
            </Link>
            <Link className="secondary-link" to="/jobs/demo-face-lab">
              {t("showcase.inspectJob")}
            </Link>
          </div>
        </div>
        <div className="proof-card panel">
          <Badge status="verified" />
          <strong>{demoEvidence.environment.gpu}</strong>
          <span>
            CUDA {demoEvidence.environment.cuda} · PyTorch{" "}
            {demoEvidence.environment.pytorch}
          </span>
          <span>
            {demoEvidence.verification.valid_models}/7{" "}
            {t("showcase.modelsVerified")}
          </span>
          <code>{demoEvidence.verification.model_root}</code>
        </div>
      </section>

      <section
        className="model-proof-strip"
        aria-label={t("showcase.modelProof")}
      >
        {["Real-ESRGAN", "GFPGAN", "CodeFormer", "RIFE v4.25"].map((model) => (
          <div key={model}>
            <i />
            <strong>{model}</strong>
            <span>{t("showcase.cudaValidated")}</span>
          </div>
        ))}
      </section>

      <section className="showcase-section">
        <div className="section-heading">
          <p className="eyebrow">{t("showcase.realResults")}</p>
          <h2>{t("showcase.fourWorkflows")}</h2>
        </div>
        <div className="scenario-grid">
          {demoScenarios.map((scenario) => (
            <article className="scenario-card panel" key={scenario.id}>
              {scenario.kind === "image" ? (
                <img
                  loading="eager"
                  src={scenario.outputs[scenario.default_output]}
                  alt={localized(scenario.title, i18n.language)}
                />
              ) : (
                <video
                  src={scenario.outputs[scenario.default_output]}
                  muted
                  loop
                  playsInline
                  preload="metadata"
                />
              )}
              <div>
                <Badge status="gpu-verified" />
                <h3>{localized(scenario.title, i18n.language)}</h3>
                <p>{localized(scenario.description, i18n.language)}</p>
                <div className="metric-pills">
                  <span>{String(scenario.metrics.output)}</span>
                  <span>{scenario.metrics.elapsed_ms} ms</span>
                </div>
                <Link to={`/workspace?scenario=${scenario.id}`}>
                  {t("showcase.tryFlow")} →
                </Link>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="showcase-section benchmark-section">
        <div className="section-heading">
          <p className="eyebrow">{t("showcase.benchmark")}</p>
          <h2>{t("showcase.measuredNotEstimated")}</h2>
          <p>{t("showcase.benchmarkMethod")}</p>
        </div>
        <div className="panel benchmark-table" role="table">
          <div className="benchmark-head" role="row">
            <strong>{t("showcase.pipeline")}</strong>
            <strong>{t("showcase.coldRun")}</strong>
            <strong>{t("showcase.warmMedian")}</strong>
            <strong>{t("showcase.peakVram")}</strong>
          </div>
          {demoEvidence.benchmarks.runs.map((run) => (
            <div role="row" key={run.id}>
              <span>{run.id}</span>
              <span>{run.cold_ms} ms</span>
              <span>
                {run.warm_median_ms === null
                  ? t("showcase.contended")
                  : `${run.warm_median_ms} ms`}
              </span>
              <span>{run.peak_vram_mib} MiB</span>
            </div>
          ))}
        </div>
      </section>

      <section className="showcase-section architecture-section">
        <div className="section-heading">
          <p className="eyebrow">{t("showcase.architecture")}</p>
          <h2>{t("showcase.productPipeline")}</h2>
        </div>
        <div
          className="architecture-flow"
          aria-label={t("showcase.productPipeline")}
        >
          {[
            "React",
            "FastAPI",
            "PostgreSQL\nOutbox",
            "Redis / RQ",
            "GPU Worker",
            "Artifact",
          ].map((node, index) => (
            <div key={node}>
              <span>{node}</span>
              {index < 5 ? <b aria-hidden="true">→</b> : null}
            </div>
          ))}
        </div>
        <div className="decision-grid">
          {decisions.map(([title, copy]) => (
            <article className="panel" key={title}>
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
      </section>

      <footer className="portfolio-footer">
        <span>Portfolio build · {buildSha.slice(0, 12)}</span>
        <a href="https://github.com/Justin21523/restoreAI-studio">GitHub</a>
        <a href="https://github.com/Justin21523/restoreAI-studio/blob/main/README.md">
          Case study
        </a>
      </footer>
    </main>
  );
}

function DemoPanel({
  scenario,
  progress,
}: {
  scenario: DemoScenario;
  progress: number;
}) {
  const { i18n, t } = useTranslation();
  const completedEvents = scenario.events.filter(
    (item) => item.progress <= progress,
  );
  return (
    <div className="demo-panel">
      <div className="demo-copy">
        <Badge status="recorded-gpu" />
        <h2>{localized(scenario.title, i18n.language)}</h2>
        <p>{localized(scenario.description, i18n.language)}</p>
      </div>
      {scenario.kind === "image" ? (
        Object.keys(scenario.outputs).length > 1 ? (
          <ModelComparison before={scenario.input} outputs={scenario.outputs} />
        ) : (
          <ImageComparison
            before={scenario.input}
            after={scenario.outputs[scenario.default_output]}
          />
        )
      ) : (
        <VideoComparison
          before={scenario.input}
          after={scenario.outputs[scenario.default_output]}
          beforeLabel={String(scenario.metrics.input)}
          afterLabel={String(scenario.metrics.output)}
        />
      )}
      <div className="pipeline-progress">
        <i style={{ width: `${progress * 100}%` }} />
      </div>
      <div className="stage-replay" aria-live="polite">
        {completedEvents.map((event) => (
          <div key={event.id}>
            <i />
            <span>{event.stage.replaceAll("_", " ")}</span>
            <small>{event.message}</small>
            <b>{event.duration_ms} ms</b>
          </div>
        ))}
      </div>
      <p className="fine-print">{t("workspace.demoNote")}</p>
      <JsonGrid
        value={{
          model: scenario.model,
          ...scenario.parameters,
          ...scenario.metrics,
        }}
      />
      {progress === 1 ? (
        <Link
          className="primary-link demo-detail-link"
          to={`/jobs/${scenario.job_id}`}
        >
          {t("workspace.inspectArtifact")}
        </Link>
      ) : null}
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
  const [strength, setStrength] = useState("0.8");
  const [fidelity, setFidelity] = useState("0.7");
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
      if ("total_items" in result) navigate(`/batches/${result.id}`);
      else navigate(`/jobs/${result.id}`);
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
      setDemoProgress(0);
      for (const item of scenario.events) {
        await new Promise((resolve) => window.setTimeout(resolve, 260));
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
    if (kind === "image") {
      form.append("face_method", faceMethod);
      form.append("strength", strength);
      form.append("fidelity", fidelity);
    }
    if (kind === "video" && operation !== "upscale")
      form.append("target_fps", targetFps);
    mutation.mutate(form);
  }
  const running = mutation.isPending || (demoProgress > 0 && demoProgress < 1);
  const faceOperation = kind === "image" && operation !== "upscale";
  const scaleOperation =
    operation.includes("upscale") || operation === "upscale";
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
                {demoScenarios.map((item) => (
                  <option value={item.id} key={item.id}>
                    {localized(item.title, i18n.language)}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <>
              <div className="segmented" aria-label={t("workspace.mediaType")}>
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
              {files.length ? (
                <ul className="selected-files">
                  {files.map((file) => (
                    <li key={`${file.name}-${file.size}`}>
                      <span>{file.name}</span>
                      <small>{formatBytes(file.size)}</small>
                    </li>
                  ))}
                </ul>
              ) : null}
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
                {scaleOperation ? (
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
                ) : null}
                {faceOperation ? (
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
                {faceOperation && faceMethod === "codeformer" ? (
                  <label>
                    {t("workspace.fidelity")} · {fidelity}
                    <input
                      aria-label={t("workspace.fidelity")}
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={fidelity}
                      onChange={(event) => setFidelity(event.target.value)}
                    />
                  </label>
                ) : null}
                {faceOperation && faceMethod === "gfpgan" ? (
                  <label>
                    {t("workspace.strength")} · {strength}
                    <input
                      aria-label={t("workspace.strength")}
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={strength}
                      onChange={(event) => setStrength(event.target.value)}
                    />
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
  const { t } = useTranslation();
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    enabled: appMode === "real",
    refetchInterval: 1500,
  });
  const demoRows = [...demoJobs, failedDemoJob];
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <p className="eyebrow">{t("jobs.eyebrow")}</p>
          <h1>{t("jobs.title")}</h1>
        </div>
        {jobs.isFetching && <span>{t("jobs.refreshing")}</span>}
      </div>
      {appMode === "demo" ? (
        <Link className="batch-callout panel" to={`/batches/${demoBatch.id}`}>
          <span>
            <Badge status="partial-failure" />{" "}
            <strong>{t("jobs.batchReplay")}</strong>
          </span>
          <span>3 items · 2 succeeded · 1 failed →</span>
        </Link>
      ) : null}
      <section className="panel job-list">
        {appMode === "demo" ? (
          demoRows.map((job) => (
            <article className="job-row" key={job.id}>
              <div>
                <Link to={`/jobs/${job.id}`}>
                  <strong>{job.original_filename}</strong>
                </Link>
                <span>
                  {job.operation.replaceAll("_", " ")} · {job.stage}
                </span>
              </div>
              <div className="job-progress">
                <i style={{ width: `${job.progress * 100}%` }} />
              </div>
              <Badge status={job.status} />
              <Link className="small-button" to={`/jobs/${job.id}`}>
                {t("jobs.inspect")}
              </Link>
              {job.error_code ? (
                <small className="job-error">
                  {job.error_code} · {job.error_message}
                </small>
              ) : null}
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
  const [retryCount, setRetryCount] = useState(0);
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
  const item =
    appMode === "demo"
      ? allDemoJobs.find((candidate) => candidate.id === jobId)
      : job.data;
  if (appMode === "real" && job.isLoading)
    return (
      <main className="page">
        <p>{t("common.loading")}</p>
      </main>
    );
  if (!item || (appMode === "real" && job.isError))
    return (
      <main className="page">
        <p className="error">{job.error?.message ?? t("common.noData")}</p>
      </main>
    );
  const artifact = item.artifact;
  const scenario = demoScenarioForJob(item.id);
  const before = scenario?.input ?? api.input(item.id);
  const after =
    scenario?.outputs[scenario.default_output] ??
    (artifact ? api.download(artifact.id) : "");
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
      {appMode === "demo" ? (
        <div className="replay-notice">
          <Badge status="recorded-replay" /> {t("detail.replayNotice")}
        </div>
      ) : null}
      {artifact && after ? (
        item.kind === "image" ? (
          scenario && Object.keys(scenario.outputs).length > 1 ? (
            <ModelComparison before={before} outputs={scenario.outputs} />
          ) : (
            <ImageComparison before={before} after={after} />
          )
        ) : (
          <VideoComparison
            before={before}
            after={after}
            beforeLabel={String(scenario?.metrics.input ?? t("common.input"))}
            afterLabel={String(scenario?.metrics.output ?? t("common.output"))}
          />
        )
      ) : (
        <section className="panel empty">
          <strong>{item.error_code}</strong>
          <p>{item.error_message ?? t("detail.noPreview")}</p>
        </section>
      )}
      <div className="detail-actions">
        {appMode === "demo" && item.status === "failed" ? (
          <button onClick={() => setRetryCount((value) => value + 1)}>
            {t("common.retry")}
          </button>
        ) : null}
        {appMode === "real" &&
        ["queued", "running", "cancelling"].includes(item.status) ? (
          <button onClick={() => action.mutate("cancel")}>
            {t("common.cancel")}
          </button>
        ) : null}
        {appMode === "real" && ["failed", "cancelled"].includes(item.status) ? (
          <button onClick={() => action.mutate("retry")}>
            {t("common.retry")}
          </button>
        ) : null}
        {artifact && after ? (
          <a className="primary-link" href={after} download>
            {t("common.download")}
          </a>
        ) : null}
        {appMode === "real" && artifact ? (
          <button className="danger" onClick={() => action.mutate("delete")}>
            {t("common.delete")}
          </button>
        ) : null}
      </div>
      {retryCount ? (
        <div className="retry-receipt panel">
          <Badge status="failed" />
          <strong>demo-retry-invalid-{retryCount}</strong>
          <span>retry_of_job_id · {item.id}</span>
          <span>INVALID_IMAGE remains actionable and auditable.</span>
        </div>
      ) : null}
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
                models: Object.keys(artifact.model_snapshot).join(", "),
                ...artifact.metadata_json,
              }}
            />
          </section>
        ) : null}
      </div>
      <section className="panel detail-card">
        <h2>{t("detail.timeline")}</h2>
        <Timeline
          events={item.events ?? []}
          liveUrl={appMode === "real" ? api.events(item.id) : undefined}
        />
      </section>
    </main>
  );
}

export function BatchDetail() {
  const { t } = useTranslation();
  const { batchId = "" } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [retried, setRetried] = useState(false);
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
  const item = appMode === "demo" ? demoBatch : batch.data;
  if (!item)
    return (
      <main className="page">
        <p className={batch.isError ? "error" : "empty"}>
          {batch.error?.message ?? t("common.loading")}
        </p>
      </main>
    );
  const progress = item.total_items
    ? (item.succeeded_items + item.failed_items) / item.total_items
    : 0;
  return (
    <main className="page">
      <div className="page-title">
        <div>
          <Link to="/jobs">← {t("common.back")}</Link>
          <p className="eyebrow">{t("detail.batch")}</p>
          <h1>{item.id}</h1>
        </div>
        <Badge status={item.status} />
      </div>
      {appMode === "demo" ? (
        <div className="replay-notice">
          <Badge status="recorded-replay" /> {t("detail.replayNotice")}
        </div>
      ) : null}
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
          {appMode === "demo" && item.failed_items ? (
            <button onClick={() => setRetried(true)}>
              {t("detail.failedOnly")}
            </button>
          ) : null}
          {appMode === "real" &&
          ["queued", "running", "cancelling"].includes(item.status) ? (
            <button onClick={() => action.mutate("cancel")}>
              {t("common.cancel")}
            </button>
          ) : null}
          {appMode === "real" && item.failed_items > 0 ? (
            <button onClick={() => action.mutate("retry")}>
              {t("detail.failedOnly")}
            </button>
          ) : null}
        </div>
      </section>
      {retried ? (
        <section className="panel retry-batch">
          <Badge status="failed" />
          <h2>demo-batch-retry-01</h2>
          <JsonGrid
            value={{
              retry_of_batch_id: item.id,
              total: 1,
              failed: 1,
              error_code: "INVALID_IMAGE",
            }}
          />
        </section>
      ) : null}
      <section className="panel job-list">
        {item.jobs?.map((job) =>
          appMode === "demo" ? (
            <article className="demo-job" key={job.id}>
              <div>
                <Badge status={job.status} />
                <h2>{job.original_filename}</h2>
                <p>{job.error_code ?? job.operation}</p>
              </div>
              <Link className="small-button" to={`/jobs/${job.id}`}>
                {t("jobs.inspect")}
              </Link>
            </article>
          ) : (
            <JobRow key={job.id} job={job} />
          ),
        )}
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
  const demoModels: ModelStatus[] = demoEvidence.models.map((model) => ({
    ...model,
    available: true,
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
        <Badge
          status={`${rows.filter((item) => item.valid).length}-verified`}
        />
      </div>
      <section className="model-grid">
        {rows.map((model) => (
          <article className="panel model-card" key={model.model_id}>
            <Badge status={model.valid ? "sha-verified" : "invalid"} />
            <h2>{model.model_id}</h2>
            <p>{model.family}</p>
            <code title={model.path}>{model.path}</code>
            <small>{formatBytes(model.size_bytes)}</small>
            {model.sha256 ? (
              <small className="model-sha">sha256 · {model.sha256}</small>
            ) : null}
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
    database: {
      status: "recorded-ready",
      role: "authoritative Job store",
      engine: "PostgreSQL 16",
    },
    redis: {
      status: "recorded-ready",
      role: "RQ dispatch",
      persistence: "not source of truth",
    },
    queue: { name: "restorai-gpu", concurrency: 1, policy: "single GPU owner" },
    worker: {
      status: "recorded-online",
      inference: "forked RQ child",
      parent_cuda: false,
    },
    gpu: demoEvidence.environment,
    storage: {
      mode: "local artifact storage",
      retention_hours: 24,
      provenance: "input/output SHA-256",
    },
    models: demoEvidence.verification,
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
          {appMode === "demo" ? (
            <p className="recorded-at">
              {t("system.recordedAt")} ·{" "}
              {new Date(demoEvidence.generated_at).toLocaleString()}
            </p>
          ) : null}
        </div>
        <Badge status={data.status} />
      </div>
      {appMode === "demo" ? (
        <div className="replay-notice">
          <Badge status="recorded-snapshot" /> {t("system.snapshotNote")}
        </div>
      ) : null}
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
