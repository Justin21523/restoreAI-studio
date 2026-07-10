/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_MODE?: "demo" | "real";
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
