import { AnalyticsWindow, METRIC_LABELS, MetricKey, WINDOW_LABELS, WINDOW_ORDER } from "./analyticsData";

export type DimKey = "models" | "api_keys";

export const TOP_N = 10;
export const TOP_MODEL_SERIES = 5;
export const RECENT_FETCH_DAYS = 2;
export const AUTO_REFRESH_MS = 15 * 60 * 1000;
export const METADATA_REFRESH_MS = 6 * 60 * 60 * 1000;

export const METRIC_OPTIONS: readonly { value: MetricKey; label: string }[] = (
  Object.keys(METRIC_LABELS) as MetricKey[]
).map((key) => ({ value: key, label: METRIC_LABELS[key] }));

export const DIM_OPTIONS: readonly { value: DimKey; label: string }[] = [
  { value: "models", label: "Models" },
  { value: "api_keys", label: "API Keys" },
];

export const WINDOW_OPTIONS: readonly { value: AnalyticsWindow; label: string }[] = WINDOW_ORDER.map((key) => ({
  value: key,
  label: WINDOW_LABELS[key],
}));

export const PALETTE = {
  spend: "#10b981",
  input: "#6366f1",
  output: "#f59e0b",
  ok: "#10b981",
  fail: "#ef4444",
  cacheRead: "#10b981",
  cacheWrite: "#06b6d4",
} as const;

export const SERIES_COLORS = [
  "#10b981",
  "#6366f1",
  "#f59e0b",
  "#0ea5e9",
  "#f43f5e",
  "#06b6d4",
  "#8b5cf6",
  "#f97316",
  "#84cc16",
  "#d946ef",
] as const;
