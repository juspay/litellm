"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChartOutlined,
  ClockCircleOutlined,
  CompressOutlined,
  ExpandOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  AnalyticsSkeleton,
  BudgetBar,
  CacheTable,
  FailureTable,
  HeatmapTable,
  IconButton,
  MetricsTable,
  SectionPanel,
  Stat,
  ToggleGroup,
  ToggleSwitch,
  rateColor,
  rateText,
} from "./analyticsPanels";
import AnalyticsGraphs from "./analyticsGraphs";
import { useAnalyticsData } from "./useAnalyticsData";
import {
  AnalyticsWindow,
  LabelRow,
  MetricKey,
  aggregateBreakdown,
  aggregateTeams,
  buildHeatmap,
  buildWallboardData,
  cacheHitRate,
  costPerMillionTokens,
  dateTotals,
  formatCount,
  formatDollars,
  formatSpend,
  periodDeltas,
  projectThirtyDay,
  selectWindow,
  shiftDay,
  successRate,
  utcDayString,
  WINDOW_DAYS,
} from "./analyticsData";
import { AUTO_REFRESH_MS, DIM_OPTIONS, DimKey, METRIC_OPTIONS, TOP_N, WINDOW_OPTIONS } from "./analyticsConfig";

export default function AnalyticsView() {
  const { accessToken } = useAuthorized();
  const [windowKey, setWindowKey] = useState<AnalyticsWindow>("30d");
  const [metric, setMetric] = useState<MetricKey>("spend");
  const [keyMetric, setKeyMetric] = useState<MetricKey>("spend");
  const [cacheDim, setCacheDim] = useState<DimKey>("models");
  const [failDim, setFailDim] = useState<DimKey>("models");
  const [showGraphs, setShowGraphs] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const { days, keyInfo, teamNames, updatedAt, loading, error, refresh } = useAnalyticsData(
    accessToken,
    windowKey,
    autoRefresh ? AUTO_REFRESH_MS : null,
  );

  useEffect(() => {
    if (!isFullscreen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsFullscreen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [isFullscreen]);

  const today = utcDayString(updatedAt ?? new Date());
  const windowed = useMemo(() => selectWindow(days, windowKey, today), [days, windowKey, today]);
  const board = useMemo(() => buildWallboardData(windowed, TOP_N), [windowed]);
  const totals = useMemo(() => dateTotals(windowed), [windowed]);
  const projected = useMemo(() => projectThirtyDay(totals, WINDOW_DAYS[windowKey]), [totals, windowKey]);
  const prevWindowed = useMemo(
    () => selectWindow(days, windowKey, shiftDay(today, -WINDOW_DAYS[windowKey])),
    [days, windowKey, today],
  );
  const deltas = useMemo(() => periodDeltas(totals, dateTotals(prevWindowed)), [totals, prevWindowed]);
  const keyRows = useMemo(() => aggregateBreakdown(windowed, "api_keys"), [windowed]);
  const teamRows = useMemo(() => aggregateTeams(windowed), [windowed]);
  const modelGroupRows = useMemo(() => aggregateBreakdown(windowed, "model_groups"), [windowed]);
  const keyHeatmap = useMemo(() => buildHeatmap(windowed, "api_keys", TOP_N), [windowed]);
  const cacheRows = useMemo(() => aggregateBreakdown(windowed, cacheDim), [windowed, cacheDim]);
  const failureRows = useMemo(() => aggregateBreakdown(windowed, failDim), [windowed, failDim]);
  const cacheRate = cacheHitRate(totals);
  const perTokenCost = costPerMillionTokens(totals);

  const shortHash = (id: string): string => (id.length > 12 ? `${id.slice(0, 8)}...` : id);
  const keyLabel = (row: LabelRow): string => keyInfo.get(row.id)?.alias ?? row.alias ?? shortHash(row.id);
  const teamLabel = (row: LabelRow): string =>
    teamNames.get(row.id) ?? row.alias ?? (row.id === "Unassigned" ? row.id : shortHash(row.id));
  const idLabel = (row: LabelRow): string => row.alias ?? row.id;
  const toggleFullscreen = () => setIsFullscreen((value) => !value);

  return (
    <div
      className={
        isFullscreen
          ? "fixed inset-0 z-50 overflow-y-auto bg-white px-[50px] py-6 text-gray-900"
          : "min-h-screen bg-white px-4 py-6 text-gray-900"
      }
    >
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold text-gray-900">Total Usage</h1>
        <div className="flex flex-wrap items-center gap-2">
          <ToggleGroup options={WINDOW_OPTIONS} value={windowKey} onChange={setWindowKey} />
          <span className="mx-1 h-5 w-px bg-gray-200" aria-hidden="true" />
          <IconButton title={isFullscreen ? "Exit fullscreen" : "Fullscreen"} onClick={toggleFullscreen}>
            {isFullscreen ? <CompressOutlined /> : <ExpandOutlined />}
          </IconButton>
          <IconButton title="Refresh now" onClick={refresh} spin={loading}>
            <ReloadOutlined />
          </IconButton>
          <ToggleSwitch
            checked={autoRefresh}
            onChange={setAutoRefresh}
            label="15m"
            title="Auto-refresh every 15 minutes"
            icon={<ClockCircleOutlined className="text-base" />}
          />
          <ToggleSwitch
            checked={showGraphs}
            onChange={setShowGraphs}
            label="Graphs"
            title="Graph Toggle"
            icon={<BarChartOutlined className="text-base" />}
          />
        </div>
      </div>

      {error ? <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-5 text-red-700">{error}</div> : null}

      {loading ? (
        <AnalyticsSkeleton variant={showGraphs ? "graphs" : "tables"} />
      ) : (
        <>
          <div className="mb-4 grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
            <Stat
              label="Total Tokens"
              value={formatCount(totals.total_tokens)}
              sub={`in ${formatCount(totals.prompt_tokens)} / out ${formatCount(totals.completion_tokens)}`}
              delta={deltas.total_tokens}
            />
            <Stat
              label="Total Spend"
              value={formatSpend(totals.spend)}
              sub={perTokenCost === null ? undefined : `${formatDollars(perTokenCost)} / 1M tokens`}
              delta={deltas.spend}
            />
            <Stat
              label="Requests"
              value={formatCount(totals.api_requests)}
              sub={`ok ${formatCount(totals.successful_requests)} / fail ${formatCount(totals.failed_requests)}`}
              delta={deltas.api_requests}
            />
            <Stat
              label="Cache"
              value={formatCount(totals.cache_read_input_tokens + totals.cache_creation_input_tokens)}
              sub={`read ${formatCount(totals.cache_read_input_tokens)} / write ${formatCount(totals.cache_creation_input_tokens)}`}
              delta={deltas.cache}
            />
            <Stat
              label="Cache Hit Rate"
              value={cacheRate === null ? "-" : `${cacheRate.toFixed(1)}%`}
              sub={`${formatCount(totals.cache_read_input_tokens)} served from cache`}
            />
            <Stat
              label="Projected 30d"
              value={formatCount(projected.total_tokens)}
              sub={`~${formatSpend(projected.spend)} spend`}
            />
          </div>

          {showGraphs ? (
            <AnalyticsGraphs results={windowed} />
          ) : (
            <>
              <div className="mb-4 grid gap-4 xl:grid-cols-2">
                <SectionPanel title={`Top ${TOP_N} Models by Spend`}>
                  <table className="mt-3 w-full text-sm tabular-nums">
                    <thead>
                      <tr className="text-left text-gray-500">
                        <th className="py-1.5 pr-2 font-medium">Model</th>
                        <th className="py-1.5 pr-2 text-right font-medium">Spend</th>
                        <th className="py-1.5 pr-2 text-right font-medium">$/1M tok</th>
                        <th className="py-1.5 pr-2 text-right font-medium">Input</th>
                        <th className="py-1.5 pr-2 text-right font-medium">Output</th>
                        <th className="py-1.5 pr-2 text-right font-medium">Requests</th>
                        <th className="py-1.5 text-right font-medium">OK%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {board.rows.map((row) => {
                        const cost = costPerMillionTokens(row.metrics);
                        return (
                          <tr key={row.id} className="border-t border-gray-200">
                            <td className="max-w-[240px] truncate py-1.5 pr-2 text-gray-900" title={row.id}>
                              {row.id}
                            </td>
                            <td className="py-1.5 pr-2 text-right text-gray-900">{formatSpend(row.metrics.spend)}</td>
                            <td className="py-1.5 pr-2 text-right text-gray-500">
                              {cost === null ? "-" : formatDollars(cost)}
                            </td>
                            <td className="py-1.5 pr-2 text-right text-gray-500">
                              {formatCount(row.metrics.prompt_tokens)}
                            </td>
                            <td className="py-1.5 pr-2 text-right text-gray-500">
                              {formatCount(row.metrics.completion_tokens)}
                            </td>
                            <td className="py-1.5 pr-2 text-right text-gray-500">
                              {formatCount(row.metrics.api_requests)}
                            </td>
                            <td className={`py-1.5 text-right ${rateColor(successRate(row.metrics))}`}>
                              {rateText(successRate(row.metrics))}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </SectionPanel>

                <SectionPanel
                  title="Usage by Day and Model"
                  right={<ToggleGroup options={METRIC_OPTIONS} value={metric} onChange={setMetric} />}
                >
                  <HeatmapTable
                    data={board}
                    metric={metric}
                    labelOf={idLabel}
                    nameHeader="Model"
                    footerLabel="All Models"
                  />
                </SectionPanel>
              </div>

              <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
                <SectionPanel title="Top API Keys by Spend">
                  <MetricsTable
                    rows={keyRows}
                    labelOf={keyLabel}
                    extraHeader="Budget"
                    extra={(row) => (
                      <BudgetBar spend={row.metrics.spend} maxBudget={keyInfo.get(row.id)?.maxBudget ?? null} />
                    )}
                  />
                </SectionPanel>
                <SectionPanel title="Spend by Team">
                  <MetricsTable rows={teamRows} labelOf={teamLabel} />
                </SectionPanel>
              </div>

              <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
                <SectionPanel title="Model Group Rollup">
                  <MetricsTable rows={modelGroupRows} labelOf={idLabel} />
                </SectionPanel>
                <SectionPanel
                  title="Usage by Day and Key"
                  right={<ToggleGroup options={METRIC_OPTIONS} value={keyMetric} onChange={setKeyMetric} />}
                >
                  <HeatmapTable data={keyHeatmap} metric={keyMetric} labelOf={keyLabel} footerLabel="All Keys" />
                </SectionPanel>
              </div>

              <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
                <SectionPanel
                  title="Cache Leaderboard"
                  right={<ToggleGroup options={DIM_OPTIONS} value={cacheDim} onChange={setCacheDim} />}
                >
                  <CacheTable rows={cacheRows} labelOf={cacheDim === "api_keys" ? keyLabel : idLabel} />
                </SectionPanel>
                <SectionPanel
                  title="Failures (worst first)"
                  right={<ToggleGroup options={DIM_OPTIONS} value={failDim} onChange={setFailDim} />}
                >
                  <FailureTable rows={failureRows} labelOf={failDim === "api_keys" ? keyLabel : idLabel} />
                </SectionPanel>
              </div>
            </>
          )}
        </>
      )}

      <div className="text-xs text-gray-500">
        {updatedAt === null
          ? "Loading usage data"
          : `Last updated ${updatedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ${windowed.length} day(s) through ${today}`}
      </div>
    </div>
  );
}
