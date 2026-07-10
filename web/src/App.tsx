import { useTranslation } from "react-i18next";
import { Link, Navigate, Route, Routes } from "react-router-dom";
import { LanguageToggle } from "./components";
import {
  appMode,
  BatchDetail,
  History,
  JobDetail,
  Models,
  Showcase,
  SystemPage,
  Workspace,
} from "./pages";

export default function App() {
  const { t } = useTranslation();
  return (
    <>
      <header>
        <Link className="brand" to="/">
          RestorAI <span>Studio</span>
        </Link>
        <nav>
          <Link to="/">{t("nav.showcase")}</Link>
          <Link to="/workspace">{t("nav.workspace")}</Link>
          <Link to="/jobs">{t("nav.jobs")}</Link>
          <Link to="/models">{t("nav.models")}</Link>
          <Link to="/system">{t("nav.system")}</Link>
        </nav>
        <LanguageToggle />
        <span className={`environment environment-${appMode}`}>{appMode}</span>
      </header>
      <Routes>
        <Route path="/" element={<Showcase />} />
        <Route path="/workspace" element={<Workspace />} />
        <Route path="/jobs" element={<History />} />
        <Route path="/jobs/:jobId" element={<JobDetail />} />
        <Route path="/batches/:batchId" element={<BatchDetail />} />
        <Route path="/models" element={<Models />} />
        <Route path="/system" element={<SystemPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}
