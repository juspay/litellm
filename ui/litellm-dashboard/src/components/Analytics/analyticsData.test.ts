import { describe, it, expect } from "vitest";
import { DailyData, SpendMetrics } from "../UsagePage/types";
import {
  aggregateBreakdown,
  aggregateTeams,
  buildDailySeries,
  buildHeatmap,
  buildTopModelSeries,
  buildWallboardData,
  cacheHitRate,
  costPerMillionTokens,
  coversRange,
  dateTotals,
  failRate,
  formatCount,
  formatSpend,
  metricValue,
  periodDeltas,
  projectThirtyDay,
  selectWindow,
  shiftDay,
  successRate,
  sumMetrics,
  utcDayString,
  windowFetchRange,
  ZERO_METRICS,
} from "./analyticsData";

const metrics = (overrides: Partial<SpendMetrics>): SpendMetrics => ({ ...ZERO_METRICS, ...overrides });

const day = (date: string, models: Record<string, SpendMetrics>): DailyData => ({
  date,
  metrics: Object.values(models).reduce(sumMetrics, ZERO_METRICS),
  breakdown: {
    models: Object.fromEntries(
      Object.entries(models).map(([model, value]) => [model, { metrics: value, metadata: {}, api_key_breakdown: {} }]),
    ),
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  },
});

const results = [
  day("2026-10-19", {
    "gpt-4o": metrics({ spend: 3, total_tokens: 300, api_requests: 3, successful_requests: 2, failed_requests: 1 }),
    "claude-sonnet": metrics({ spend: 2, total_tokens: 200, api_requests: 2, successful_requests: 2 }),
    "gemini-flash": metrics({ spend: 1, total_tokens: 100, api_requests: 1 }),
  }),
  day("2026-10-18", {
    "gpt-4o": metrics({ spend: 1, total_tokens: 100, api_requests: 1 }),
  }),
  day("2026-10-15", {
    "gpt-4o": metrics({ spend: 5, total_tokens: 500, api_requests: 5 }),
  }),
  day("2026-09-30", {
    "gpt-4o": metrics({ spend: 99, total_tokens: 9900, api_requests: 9 }),
  }),
];

describe("coversRange and windowFetchRange", () => {
  it("spans the window plus its delta baseline ending at the window end", () => {
    const today = utcDayString(new Date());
    expect(windowFetchRange("today")).toEqual([shiftDay(today, -1), today]);
    expect(windowFetchRange("yesterday")).toEqual([shiftDay(today, -2), shiftDay(today, -1)]);
    expect(windowFetchRange("5d")).toEqual([shiftDay(today, -9), today]);
    expect(windowFetchRange("30d")).toEqual([shiftDay(today, -59), today]);
  });

  it("treats a cached range as covering equal or narrower wanted ranges only", () => {
    expect(coversRange(null, ["2026-10-01", "2026-10-20"])).toBe(false);
    expect(coversRange(["2026-09-01", "2026-10-20"], ["2026-10-01", "2026-10-20"])).toBe(true);
    expect(coversRange(["2026-10-01", "2026-10-20"], ["2026-09-01", "2026-10-20"])).toBe(false);
    expect(coversRange(["2026-10-01", "2026-10-19"], ["2026-10-01", "2026-10-20"])).toBe(false);
    expect(coversRange(["2026-10-01", "2026-10-20"], ["2026-10-01", "2026-10-20"])).toBe(true);
  });
});

describe("selectWindow", () => {
  it("slices today and yesterday to single days", () => {
    expect(selectWindow(results, "today", "2026-10-19").map((r) => r.date)).toEqual(["2026-10-19"]);
    expect(selectWindow(results, "yesterday", "2026-10-19").map((r) => r.date)).toEqual(["2026-10-18"]);
  });

  it("slices trailing N-day windows ending on today and drops out-of-range days", () => {
    const windowed = selectWindow(results, "5d", "2026-10-19").map((r) => r.date);

    expect(windowed).toEqual(["2026-10-19", "2026-10-18", "2026-10-15"]);
    expect(windowed).not.toContain("2026-09-30");
  });
});

describe("utcDayString", () => {
  it("labels instants by UTC calendar day across local midnight boundaries", () => {
    expect(utcDayString(new Date("2025-12-31T19:00:00Z"))).toBe("2025-12-31");
    expect(utcDayString(new Date("2026-01-01T00:00:00Z"))).toBe("2026-01-01");
  });
});

describe("buildWallboardData", () => {
  it("ranks models by spend and keeps only the top N", () => {
    const data = buildWallboardData(selectWindow(results, "today", "2026-10-19"), 2);

    expect(data.rows.map((row) => row.id)).toEqual(["gpt-4o", "claude-sonnet"]);
  });

  it("aligns matrix cells to ascending dates and zero-fills gaps", () => {
    const data = buildWallboardData(selectWindow(results, "5d", "2026-10-19"), 5);

    expect(data.dates).toEqual(["2026-10-15", "2026-10-18", "2026-10-19"]);
    const gemini = data.rows.find((row) => row.id === "gemini-flash");
    expect(gemini?.cells.map((cell) => metricValue(cell, "spend"))).toEqual([0, 0, 1]);
  });

  it("totals cover every model even beyond the visible rows", () => {
    const windowed = selectWindow(results, "today", "2026-10-19");
    const data = buildWallboardData(windowed, 1);

    expect(data.rows.map((row) => row.id)).toEqual(["gpt-4o"]);
    expect(data.totals.spend).toBe(6);
    expect(data.totals.total_tokens).toBe(600);
    expect(data.columnTotals[0].spend).toBe(6);
    expect(dateTotals(windowed).spend).toBe(6);
  });

  it("keeps each row's cell sums equal to its range metrics", () => {
    const data = buildWallboardData(selectWindow(results, "5d", "2026-10-19"), 5);

    data.rows.forEach((row) => {
      expect(row.cells.reduce((acc, cell) => acc + metricValue(cell, "spend"), 0)).toBeCloseTo(row.metrics.spend);
    });
  });

  it("computes per-date totals across all models", () => {
    const data = buildWallboardData(selectWindow(results, "today", "2026-10-19"), 5);

    expect(data.columnTotals.map((t) => t.spend)).toEqual([6]);
    expect(data.columnTotals[0].api_requests).toBe(6);
  });
});

describe("derived values and formatting", () => {
  it("computes success rate only when requests exist", () => {
    expect(successRate(metrics({ api_requests: 4, successful_requests: 3 }))).toBe(75);
    expect(successRate(metrics({ api_requests: 0 }))).toBeNull();
  });

  it("computes per-million-token cost only when tokens exist", () => {
    expect(costPerMillionTokens(metrics({ spend: 5, total_tokens: 2_000_000 }))).toBe(2.5);
    expect(costPerMillionTokens(metrics({ spend: 5 }))).toBeNull();
  });

  it("projects a 30-day volume from the window's daily average", () => {
    const projected = projectThirtyDay(metrics({ total_tokens: 10_000, spend: 10 }), 5);

    expect(projected.total_tokens).toBe(60_000);
    expect(projected.spend).toBe(60);
  });

  it("abbreviates spend and counts with consistent precision including trillions", () => {
    expect(formatSpend(12.345)).toBe("$12.35");
    expect(formatSpend(2500)).toBe("$2.50k");
    expect(formatCount(1500)).toBe("1.50k");
    expect(formatCount(2_500_000)).toBe("2.50M");
    expect(formatCount(900_000_000_000)).toBe("900.00B");
    expect(formatCount(1_200_000_000_000)).toBe("1.20T");
    expect(formatCount(42)).toBe("42");
  });
});

describe("aggregateBreakdown", () => {
  const keyEntry = (alias: string | null, value: SpendMetrics) => ({
    metrics: value,
    metadata: { key_alias: alias, team_id: "team-1" },
  });

  it("sums entries across days and sorts by spend desc with alias metadata", () => {
    const rows = aggregateBreakdown(
      [
        day("2026-10-19", {}),
        {
          date: "2026-10-19",
          metrics: metrics({ spend: 5 }),
          breakdown: {
            models: {},
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: { hashA: keyEntry("alpha", metrics({ spend: 3, api_requests: 2 })) },
            entities: {},
          },
        },
        {
          date: "2026-10-18",
          metrics: metrics({ spend: 9 }),
          breakdown: {
            models: {},
            model_groups: {},
            mcp_servers: {},
            providers: {},
            api_keys: {
              hashA: keyEntry("alpha", metrics({ spend: 1, api_requests: 1 })),
              hashB: keyEntry(null, metrics({ spend: 7, api_requests: 5 })),
            },
            entities: {},
          },
        },
      ],
      "api_keys",
    );

    expect(rows.map((row) => [row.id, row.alias, row.metrics.spend, row.metrics.api_requests])).toEqual([
      ["hashB", null, 7, 5],
      ["hashA", "alpha", 4, 3],
    ]);
  });
});

describe("aggregateTeams", () => {
  it("groups api keys by team_id and buckets missing teams as Unassigned", () => {
    const rows = aggregateTeams([
      {
        date: "2026-10-19",
        metrics: metrics({ spend: 5 }),
        breakdown: {
          models: {},
          model_groups: {},
          mcp_servers: {},
          providers: {},
          api_keys: {
            hashA: { metrics: metrics({ spend: 3 }), metadata: { key_alias: "a", team_id: "team-1" } },
            hashB: { metrics: metrics({ spend: 2 }), metadata: { key_alias: "b", team_id: null } },
          },
          entities: {},
        },
      },
    ]);

    expect(rows.map((row) => [row.id, row.metrics.spend])).toEqual([
      ["team-1", 3],
      ["Unassigned", 2],
    ]);
  });
});

describe("cacheHitRate and failRate", () => {
  it("computes cache hit share and returns null without traffic", () => {
    expect(cacheHitRate(metrics({ cache_read_input_tokens: 75, cache_creation_input_tokens: 25 }))).toBe(75);
    expect(cacheHitRate(metrics({ spend: 1 }))).toBeNull();
  });

  it("computes failure share and returns null without requests", () => {
    expect(failRate(metrics({ api_requests: 8, failed_requests: 2 }))).toBe(25);
    expect(failRate(metrics({ spend: 1 }))).toBeNull();
  });
});

describe("shiftDay and periodDeltas", () => {
  it("shifts day strings across month boundaries", () => {
    expect(shiftDay("2026-11-02", -5)).toBe("2026-10-28");
    expect(shiftDay("2026-10-19", 0)).toBe("2026-10-19");
  });

  it("computes signed percent changes and marks fresh activity as null", () => {
    const deltas = periodDeltas(
      metrics({ spend: 150, total_tokens: 300, api_requests: 0, cache_read_input_tokens: 30 }),
      metrics({ spend: 100, total_tokens: 250, api_requests: 0, cache_creation_input_tokens: 10 }),
    );
    expect(deltas.spend).toBe(50);
    expect(deltas.total_tokens).toBe(20);
    expect(deltas.api_requests).toBe(0);
    expect(deltas.cache).toBe(200);
    expect(periodDeltas(metrics({ spend: 5 }), metrics({})).spend).toBeNull();
    expect(periodDeltas(metrics({}), metrics({})).spend).toBe(0);
  });
});

describe("buildHeatmap", () => {
  it("aligns cells to dates, truncates to topN, and drops zero-only rows below the cut", () => {
    const heatmap = buildHeatmap(selectWindow(results, "5d", "2026-10-19"), "models", 2);

    expect(heatmap.dates).toEqual(["2026-10-15", "2026-10-18", "2026-10-19"]);
    expect(heatmap.rows.map((row) => row.id)).toEqual(["gpt-4o", "claude-sonnet"]);
    expect(heatmap.rows[0].cells.map((cell) => cell.spend)).toEqual([5, 1, 3]);
  });
});

describe("buildDailySeries", () => {
  it("merges same-day entries and orders points ascending by date", () => {
    const series = buildDailySeries([
      day("2026-10-19", { "gpt-4o": metrics({ spend: 3, total_tokens: 30 }) }),
      day("2026-10-15", { "gpt-4o": metrics({ spend: 5, total_tokens: 50 }) }),
      {
        date: "2026-10-19",
        metrics: metrics({ spend: 1, total_tokens: 10 }),
        breakdown: {
          models: {
            "claude-sonnet": { metrics: metrics({ spend: 1, total_tokens: 10 }), metadata: {}, api_key_breakdown: {} },
          },
          model_groups: {},
          mcp_servers: {},
          providers: {},
          api_keys: {},
          entities: {},
        },
      },
    ]);

    expect(series.map((point) => [point.date, point.spend, point.total_tokens])).toEqual([
      ["2026-10-15", 5, 50],
      ["2026-10-19", 4, 40],
    ]);
  });
});

describe("buildTopModelSeries", () => {
  it("keeps topN models and aligns per-date values with zero fill", () => {
    const series = buildTopModelSeries(
      [
        day("2026-10-19", {
          "gpt-4o": metrics({ spend: 3 }),
          "claude-sonnet": metrics({ spend: 2 }),
          "gemini-flash": metrics({ spend: 1 }),
        }),
        day("2026-10-18", { "gpt-4o": metrics({ spend: 1 }) }),
      ],
      2,
    );

    expect(series.models).toEqual(["gpt-4o", "claude-sonnet"]);
    expect(series.points.map((point) => [point.date, point.values])).toEqual([
      ["2026-10-18", { "gpt-4o": 1, "claude-sonnet": 0 }],
      ["2026-10-19", { "gpt-4o": 3, "claude-sonnet": 2 }],
    ]);
  });
});
