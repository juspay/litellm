import { DailyData, SpendMetrics } from "../UsagePage/types";

export type MetricKey = "spend" | "total_tokens" | "api_requests";

export type AnalyticsWindow = "today" | "yesterday" | "5d" | "10d" | "30d";

export const METRIC_LABELS: Record<MetricKey, string> = {
  spend: "Spend",
  total_tokens: "Tokens",
  api_requests: "Requests",
};

export const WINDOW_ORDER: readonly AnalyticsWindow[] = ["today", "yesterday", "5d", "10d", "30d"];

export const WINDOW_LABELS: Record<AnalyticsWindow, string> = {
  today: "Today",
  yesterday: "Yesterday",
  "5d": "Last 5 Days",
  "10d": "Last 10 Days",
  "30d": "Last 30 Days",
};

export const WINDOW_DAYS: Record<AnalyticsWindow, number> = {
  today: 1,
  yesterday: 1,
  "5d": 5,
  "10d": 10,
  "30d": 30,
};

export const ZERO_METRICS: SpendMetrics = {
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

const DAY_MS = 86_400_000;

const parseDay = (day: string): number => Date.parse(`${day}T00:00:00Z`);

const formatDay = (ms: number): string => new Date(ms).toISOString().slice(0, 10);

export const utcDayString = (date: Date): string => formatDay(date.getTime());

export const sumMetrics = (a: SpendMetrics, b: SpendMetrics): SpendMetrics => ({
  spend: a.spend + b.spend,
  prompt_tokens: a.prompt_tokens + b.prompt_tokens,
  completion_tokens: a.completion_tokens + b.completion_tokens,
  total_tokens: a.total_tokens + b.total_tokens,
  api_requests: a.api_requests + b.api_requests,
  successful_requests: a.successful_requests + b.successful_requests,
  failed_requests: a.failed_requests + b.failed_requests,
  cache_read_input_tokens: a.cache_read_input_tokens + b.cache_read_input_tokens,
  cache_creation_input_tokens: a.cache_creation_input_tokens + b.cache_creation_input_tokens,
});

export const coversRange = (cached: readonly [string, string] | null, wanted: readonly [string, string]): boolean =>
  cached !== null && cached[0] <= wanted[0] && cached[1] >= wanted[1];

export const windowFetchRange = (win: AnalyticsWindow): readonly [string, string] => {
  const end = shiftDay(utcDayString(new Date()), win === "yesterday" ? -1 : 0);
  return [shiftDay(end, -(2 * WINDOW_DAYS[win] - 1)), end];
};

export const selectWindow = (results: DailyData[], win: AnalyticsWindow, today: string): DailyData[] => {
  const endMs = parseDay(today) - (win === "yesterday" ? DAY_MS : 0);
  const days = WINDOW_DAYS[win];
  const windowDates = new Set(Array.from({ length: days }, (_, i) => formatDay(endMs - (days - 1 - i) * DAY_MS)));
  return results.filter((result) => windowDates.has(result.date));
};

const toHeatmap = (dates: string[], sorted: HeatmapRow[], topN: number): HeatmapData => ({
  dates,
  rows: sorted.slice(0, topN),
  columnTotals: dates.map((_, index) => sorted.map((row) => row.cells[index]).reduce(sumMetrics, ZERO_METRICS)),
  totals: sorted.map((row) => row.metrics).reduce(sumMetrics, ZERO_METRICS),
});

export const buildWallboardData = (results: DailyData[], topN: number): HeatmapData => {
  const dates = [...new Set(results.map((result) => result.date))].sort();

  const byModel = new Map<string, Map<string, SpendMetrics>>();
  results.forEach((day) => {
    Object.entries(day.breakdown?.models ?? {}).forEach(([rawModel, entry]) => {
      const id = rawModel || "unknown";
      const perDate = byModel.get(id) ?? new Map<string, SpendMetrics>();
      perDate.set(day.date, sumMetrics(perDate.get(day.date) ?? ZERO_METRICS, entry.metrics));
      byModel.set(id, perDate);
    });
  });

  const sorted = [...byModel.entries()]
    .map(
      ([id, perDate]): HeatmapRow => ({
        id,
        alias: null,
        metrics: [...perDate.values()].reduce(sumMetrics, ZERO_METRICS),
        cells: dates.map((date) => perDate.get(date) ?? ZERO_METRICS),
      }),
    )
    .sort((a, b) => b.metrics.spend - a.metrics.spend || a.id.localeCompare(b.id));

  return toHeatmap(dates, sorted, topN);
};

export const dateTotals = (results: DailyData[]): SpendMetrics =>
  results.map((result) => result.metrics).reduce(sumMetrics, ZERO_METRICS);

export const metricValue = (metrics: SpendMetrics, key: MetricKey): number => metrics[key];

export const successRate = (metrics: SpendMetrics): number | null =>
  metrics.api_requests > 0 ? (metrics.successful_requests / metrics.api_requests) * 100 : null;

export const costPerMillionTokens = (metrics: SpendMetrics): number | null =>
  metrics.total_tokens > 0 ? (metrics.spend / metrics.total_tokens) * 1_000_000 : null;

export const projectThirtyDay = (metrics: SpendMetrics, dayCount: number): SpendMetrics => {
  if (dayCount <= 0) return ZERO_METRICS;
  const factor = 30 / dayCount;
  return {
    spend: metrics.spend * factor,
    prompt_tokens: metrics.prompt_tokens * factor,
    completion_tokens: metrics.completion_tokens * factor,
    total_tokens: metrics.total_tokens * factor,
    api_requests: metrics.api_requests * factor,
    successful_requests: metrics.successful_requests * factor,
    failed_requests: metrics.failed_requests * factor,
    cache_read_input_tokens: metrics.cache_read_input_tokens * factor,
    cache_creation_input_tokens: metrics.cache_creation_input_tokens * factor,
  };
};

export const formatSpend = (value: number): string => {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `$${(value / 1_000).toFixed(2)}k`;
  return `$${value.toFixed(2)}`;
};

export const formatDollars = (value: number): string => `$${value.toFixed(2)}`;

export const formatCount = (value: number): string => {
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000_000) return `${(value / 1_000_000_000_000).toFixed(2)}T`;
  if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(2)}k`;
  return `${value}`;
};

export const formatMetric = (metrics: SpendMetrics, key: MetricKey): string =>
  key === "spend" ? formatSpend(metrics.spend) : formatCount(metrics[key]);

export const prettyDay = (day: string): string =>
  new Date(parseDay(day)).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

export type DailySeriesPoint = SpendMetrics & { date: string };

export interface ModelSeriesPoint {
  date: string;
  values: Readonly<Record<string, number>>;
}

export interface TopModelSeries {
  models: string[];
  points: ModelSeriesPoint[];
}

export const buildDailySeries = (results: DailyData[]): DailySeriesPoint[] => {
  const byDate = new Map<string, SpendMetrics>();
  results.forEach((result) => {
    byDate.set(result.date, sumMetrics(byDate.get(result.date) ?? ZERO_METRICS, result.metrics));
  });
  return [...byDate.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([date, metrics]) => ({ ...metrics, date }));
};

export const buildTopModelSeries = (results: DailyData[], topN: number): TopModelSeries => {
  const models = aggregateBreakdown(results, "models")
    .slice(0, topN)
    .map((row) => row.id);
  const perDate = new Map<string, Map<string, number>>();
  results.forEach((result) => {
    const day = perDate.get(result.date) ?? new Map<string, number>();
    Object.entries(result.breakdown?.models ?? {}).forEach(([model, entry]) => {
      day.set(model, (day.get(model) ?? 0) + entry.metrics.spend);
    });
    perDate.set(result.date, day);
  });
  return {
    models,
    points: [...perDate.keys()].sort().map((date) => {
      const day = perDate.get(date) ?? new Map<string, number>();
      return {
        date,
        values: Object.fromEntries(models.map((model) => [model, day.get(model) ?? 0])),
      };
    }),
  };
};

export type BreakdownField =
  | "models"
  | "model_groups"
  | "providers"
  | "mcp_servers"
  | "endpoints"
  | "api_keys"
  | "entities";

export interface LabelRow {
  id: string;
  alias: string | null;
}

export interface BreakdownRow extends LabelRow {
  metrics: SpendMetrics;
}

export interface HeatmapRow extends LabelRow {
  metrics: SpendMetrics;
  cells: SpendMetrics[];
}

export interface HeatmapData {
  dates: string[];
  rows: HeatmapRow[];
  columnTotals: SpendMetrics[];
  totals: SpendMetrics;
}

export interface DeltaMetrics {
  spend: number | null;
  total_tokens: number | null;
  api_requests: number | null;
  cache: number | null;
}

const metadataLabel = (metadata: object): string | null => {
  const record = metadata as Record<string, unknown>;
  const candidate = record.key_alias ?? record.alias ?? record.user_email ?? record.team_alias ?? record.label;
  return typeof candidate === "string" && candidate.length > 0 ? candidate : null;
};

const entryAlias = (field: BreakdownField, metadata: object): string | null =>
  field === "api_keys" || field === "entities" ? metadataLabel(metadata) : null;

export const aggregateBreakdown = (results: DailyData[], field: BreakdownField): BreakdownRow[] => {
  const sums = new Map<string, BreakdownRow>();
  results.forEach((day) => {
    Object.entries(day.breakdown?.[field] ?? {}).forEach(([rawId, entry]) => {
      const id = rawId || "unknown";
      const existing = sums.get(id);
      sums.set(id, {
        id,
        alias: existing?.alias ?? entryAlias(field, entry.metadata ?? {}),
        metrics: sumMetrics(existing?.metrics ?? ZERO_METRICS, entry.metrics),
      });
    });
  });
  return [...sums.values()].sort((a, b) => b.metrics.spend - a.metrics.spend || a.id.localeCompare(b.id));
};

export const aggregateTeams = (results: DailyData[]): BreakdownRow[] => {
  const sums = new Map<string, SpendMetrics>();
  results.forEach((day) => {
    Object.values(day.breakdown?.api_keys ?? {}).forEach((entry) => {
      const teamId = entry.metadata?.team_id || "Unassigned";
      sums.set(teamId, sumMetrics(sums.get(teamId) ?? ZERO_METRICS, entry.metrics));
    });
  });
  return [...sums.entries()]
    .map(([id, metrics]) => ({ id, alias: null, metrics }))
    .sort((a, b) => b.metrics.spend - a.metrics.spend || a.id.localeCompare(b.id));
};

export const cacheHitRate = (metrics: SpendMetrics): number | null => {
  const traffic = metrics.cache_read_input_tokens + metrics.cache_creation_input_tokens;
  return traffic > 0 ? (metrics.cache_read_input_tokens / traffic) * 100 : null;
};

export const failRate = (metrics: SpendMetrics): number | null =>
  metrics.api_requests > 0 ? (metrics.failed_requests / metrics.api_requests) * 100 : null;

export const shiftDay = (day: string, days: number): string => formatDay(parseDay(day) + days * DAY_MS);

const pctChange = (current: number, previous: number): number | null => {
  if (previous === 0) return current === 0 ? 0 : null;
  return ((current - previous) / previous) * 100;
};

export const periodDeltas = (current: SpendMetrics, previous: SpendMetrics): DeltaMetrics => ({
  spend: pctChange(current.spend, previous.spend),
  total_tokens: pctChange(current.total_tokens, previous.total_tokens),
  api_requests: pctChange(current.api_requests, previous.api_requests),
  cache: pctChange(
    current.cache_read_input_tokens + current.cache_creation_input_tokens,
    previous.cache_read_input_tokens + previous.cache_creation_input_tokens,
  ),
});

export const buildHeatmap = (
  results: DailyData[],
  field: "models" | "api_keys" | "entities",
  topN: number,
): HeatmapData => {
  const dates = [...new Set(results.map((result) => result.date))].sort();
  const byId = new Map<string, { alias: string | null; perDate: Map<string, SpendMetrics> }>();
  results.forEach((day) => {
    Object.entries(day.breakdown?.[field] ?? {}).forEach(([rawId, entry]) => {
      const id = rawId || "unknown";
      const slot = byId.get(id) ?? {
        alias: entryAlias(field, entry.metadata ?? {}),
        perDate: new Map<string, SpendMetrics>(),
      };
      slot.perDate.set(day.date, sumMetrics(slot.perDate.get(day.date) ?? ZERO_METRICS, entry.metrics));
      byId.set(id, slot);
    });
  });
  const sorted = [...byId.entries()]
    .map(
      ([id, slot]): HeatmapRow => ({
        id,
        alias: slot.alias,
        metrics: [...slot.perDate.values()].reduce(sumMetrics, ZERO_METRICS),
        cells: dates.map((date) => slot.perDate.get(date) ?? ZERO_METRICS),
      }),
    )
    .sort((a, b) => b.metrics.spend - a.metrics.spend || a.id.localeCompare(b.id));
  return toHeatmap(dates, sorted, topN);
};
