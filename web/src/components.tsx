import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { api } from "./api";
import type { Job, JobEvent } from "./types";

export function Badge({ status }: { status: string }) {
  return (
    <span className={`badge badge-${status}`}>
      {status.replaceAll("_", " ")}
    </span>
  );
}

export function LanguageToggle() {
  const { i18n } = useTranslation();
  const language = i18n.language.startsWith("zh") ? "zh" : "en";
  return (
    <button
      className="language-toggle"
      type="button"
      aria-label="Switch language"
      onClick={() => void i18n.changeLanguage(language === "zh" ? "en" : "zh")}
    >
      {language === "zh" ? "EN" : "繁中"}
    </button>
  );
}

export function JobRow({ job }: { job: Job }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const action = useMutation({
    mutationFn: (type: "cancel" | "retry") =>
      type === "cancel" ? api.cancel(job.id) : api.retry(job.id),
    onSuccess: (next) => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] });
      if (next.id !== job.id) window.location.hash = `#/jobs/${next.id}`;
    },
  });
  return (
    <article className="job-row">
      <div>
        <Link to={`/jobs/${job.id}`}>
          <strong>{job.original_filename}</strong>
        </Link>
        <span>
          {job.operation.replaceAll("_", " ")} · {job.stage}
        </span>
        {job.batch_id ? (
          <Link className="subtle-link" to={`/batches/${job.batch_id}`}>
            Batch {job.batch_id.slice(0, 8)}
          </Link>
        ) : null}
      </div>
      <div
        className="job-progress"
        aria-label={`${Math.round(job.progress * 100)} percent`}
      >
        <i style={{ width: `${job.progress * 100}%` }} />
      </div>
      <Badge status={job.status} />
      <div className="row-actions">
        {job.artifact_id && job.status === "succeeded" ? (
          <a className="small-button" href={api.download(job.artifact_id)}>
            {t("common.download")}
          </a>
        ) : null}
        {["queued", "running", "cancelling"].includes(job.status) ? (
          <button
            className="small-button secondary"
            disabled={action.isPending || job.status === "cancelling"}
            onClick={() => action.mutate("cancel")}
          >
            {t("common.cancel")}
          </button>
        ) : null}
        {["failed", "cancelled"].includes(job.status) ? (
          <button
            className="small-button"
            disabled={action.isPending}
            onClick={() => action.mutate("retry")}
          >
            {t("common.retry")}
          </button>
        ) : null}
      </div>
      {job.error_message ? (
        <small className="job-error" title={job.error_message}>
          {job.error_code}
        </small>
      ) : null}
    </article>
  );
}

export function ImageComparison({
  before,
  after,
}: {
  before: string;
  after: string;
}) {
  const { t } = useTranslation();
  const [position, setPosition] = useState(50);
  const [zoom, setZoom] = useState(1);
  return (
    <div className="comparison-shell">
      <div className="comparison-toolbar">
        <button type="button" onClick={() => setZoom(Math.max(1, zoom - 0.25))}>
          −
        </button>
        <span>{Math.round(zoom * 100)}%</span>
        <button type="button" onClick={() => setZoom(Math.min(2, zoom + 0.25))}>
          +
        </button>
        <button
          type="button"
          onClick={() => {
            setZoom(1);
            setPosition(50);
          }}
        >
          100%
        </button>
      </div>
      <div className="comparison-scroll">
        <div className="comparison" style={{ transform: `scale(${zoom})` }}>
          <img src={before} alt={t("common.input")} />
          <div
            className="comparison-after"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          >
            <img src={after} alt={t("common.output")} />
          </div>
          <div className="comparison-line" style={{ left: `${position}%` }} />
          <span className="comparison-label left">{t("common.input")}</span>
          <span className="comparison-label right">{t("common.output")}</span>
          <input
            aria-label="Before and after position"
            type="range"
            min="0"
            max="100"
            value={position}
            onChange={(event) => setPosition(Number(event.target.value))}
          />
        </div>
      </div>
    </div>
  );
}

export function VideoComparison({
  before,
  after,
  beforeLabel,
  afterLabel,
}: {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const { t } = useTranslation();
  const first = useRef<HTMLVideoElement>(null);
  const second = useRef<HTMLVideoElement>(null);
  const [speed, setSpeed] = useState(1);
  function play() {
    void first.current?.play();
    void second.current?.play();
  }
  function pause() {
    first.current?.pause();
    second.current?.pause();
  }
  function sync() {
    if (
      first.current &&
      second.current &&
      Math.abs(first.current.currentTime - second.current.currentTime) > 0.15
    )
      second.current.currentTime = first.current.currentTime;
  }
  function changeSpeed(value: number) {
    setSpeed(value);
    if (first.current) first.current.playbackRate = value;
    if (second.current) second.current.playbackRate = value;
  }
  return (
    <div className="video-shell">
      <div className="playback-toolbar" aria-label={t("detail.playbackSpeed")}>
        <span>{t("detail.playbackSpeed")}</span>
        {[0.25, 0.5, 1].map((value) => (
          <button
            type="button"
            className={speed === value ? "active" : ""}
            key={value}
            onClick={() => changeSpeed(value)}
          >
            {value}×
          </button>
        ))}
      </div>
      <div className="video-comparison">
        <figure>
          <video
            ref={first}
            src={before}
            controls
            preload="metadata"
            onPlay={play}
            onPause={pause}
            onTimeUpdate={sync}
          />
          <figcaption>{beforeLabel ?? t("common.input")}</figcaption>
        </figure>
        <figure>
          <video
            ref={second}
            src={after}
            controls
            preload="metadata"
            onPlay={play}
            onPause={pause}
          />
          <figcaption>{afterLabel ?? t("common.output")}</figcaption>
        </figure>
      </div>
    </div>
  );
}

export function ModelComparison({
  before,
  outputs,
}: {
  before: string;
  outputs: Record<string, string>;
}) {
  const { t } = useTranslation();
  const choices = Object.keys(outputs);
  const [selected, setSelected] = useState(
    choices.includes("restored") ? "restored" : choices[0],
  );
  return (
    <div className="model-comparison">
      <div
        className="model-tabs"
        role="tablist"
        aria-label={t("detail.modelOutput")}
      >
        {choices.map((choice) => (
          <button
            type="button"
            role="tab"
            aria-selected={choice === selected}
            className={choice === selected ? "active" : ""}
            key={choice}
            onClick={() => setSelected(choice)}
          >
            {choice === "restored" ? t("detail.completePipeline") : choice}
          </button>
        ))}
      </div>
      <ImageComparison before={before} after={outputs[selected]} />
    </div>
  );
}

export function Timeline({
  events,
  liveUrl,
}: {
  events: JobEvent[];
  liveUrl?: string;
}) {
  const [live, setLive] = useState(events);
  useEffect(() => setLive(events), [events]);
  useEffect(() => {
    if (!liveUrl) return;
    const source = new EventSource(liveUrl);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as JobEvent;
      event.event_type = event.type ?? "progress";
      event.created_at = event.timestamp ?? new Date().toISOString();
      setLive((current) =>
        [...current.filter((item) => item.id !== event.id), event].sort(
          (a, b) => a.id - b.id,
        ),
      );
    };
    return () => source.close();
  }, [liveUrl]);
  return (
    <ol className="timeline">
      {live.map((event) => (
        <li key={event.id}>
          <i />
          <div>
            <strong>{event.stage.replaceAll("_", " ")}</strong>
            <span>{event.message}</span>
            {typeof event.payload?.duration_ms === "number" ? (
              <small>{event.payload.duration_ms} ms</small>
            ) : null}
            <time>{new Date(event.created_at).toLocaleString()}</time>
          </div>
          <b>{Math.round(event.progress * 100)}%</b>
        </li>
      ))}
    </ol>
  );
}

export function JsonGrid({ value }: { value: Record<string, unknown> }) {
  return (
    <dl className="data-grid">
      {Object.entries(value).map(([key, item]) => (
        <div key={key}>
          <dt>{key.replaceAll("_", " ")}</dt>
          <dd>
            {typeof item === "object" ? JSON.stringify(item) : String(item)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}
