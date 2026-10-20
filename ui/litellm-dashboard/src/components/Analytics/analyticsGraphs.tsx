"use client";

import { useMemo } from "react";
import { DailyData } from "../UsagePage/types";
import { TrendChart } from "./analyticsChart";
import { SectionPanel, ShareBars } from "./analyticsPanels";
import { PALETTE, SERIES_COLORS, TOP_MODEL_SERIES } from "./analyticsConfig";
import {
  DailySeriesPoint,
  aggregateBreakdown,
  buildDailySeries,
  buildTopModelSeries,
  formatCount,
  formatSpend,
  prettyDay,
} from "./analyticsData";

const GRAPHS_KEYFRAMES =
  "@keyframes usage-fade-in{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}.usage-graphs>div{animation:usage-fade-in 0.5s ease-out both}.usage-graphs>div:nth-child(2){animation-delay:0.08s}.usage-graphs>div:nth-child(3){animation-delay:0.16s}.usage-graphs>div:nth-child(4){animation-delay:0.24s}";

const chartPoint = (point: DailySeriesPoint): Record<string, number | string> => ({
  ...point,
  date: prettyDay(point.date),
});

const EmptyChart = () => (
  <div className="mt-3 flex h-56 items-center justify-center text-sm text-gray-400">No data for this period</div>
);

export default function AnalyticsGraphs({ results }: { results: DailyData[] }) {
  const daily = useMemo(() => buildDailySeries(results), [results]);
  const modelSeries = useMemo(() => buildTopModelSeries(results, TOP_MODEL_SERIES), [results]);
  const modelRows = useMemo(() => aggregateBreakdown(results, "models"), [results]);

  if (daily.length === 0) {
    return (
      <SectionPanel title="Graphs">
        <EmptyChart />
      </SectionPanel>
    );
  }

  const dailyPoints = daily.map(chartPoint);
  const modelPoints = modelSeries.points.map((point) => ({
    date: prettyDay(point.date),
    ...point.values,
  }));

  return (
    <>
      <style>{GRAPHS_KEYFRAMES}</style>
      <div className="usage-graphs">
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SectionPanel title="Spend by Day">
            <div className="mt-3">
              <TrendChart
                type="area"
                data={dailyPoints}
                index="date"
                categories={["spend"]}
                colors={[PALETTE.spend]}
                valueFormatter={formatSpend}
                showLegend={false}
              />
            </div>
          </SectionPanel>
          <SectionPanel title="Tokens by Day">
            <div className="mt-3">
              <TrendChart
                type="line"
                data={dailyPoints}
                index="date"
                categories={["prompt_tokens", "completion_tokens"]}
                colors={[PALETTE.input, PALETTE.output]}
                valueFormatter={formatCount}
              />
            </div>
          </SectionPanel>
        </div>

        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SectionPanel title="Requests by Day">
            <div className="mt-3">
              <TrendChart
                type="bar"
                data={dailyPoints}
                index="date"
                categories={["successful_requests", "failed_requests"]}
                colors={[PALETTE.ok, PALETTE.fail]}
                valueFormatter={formatCount}
                stacked
              />
            </div>
          </SectionPanel>
          <SectionPanel title="Cache Traffic by Day">
            <div className="mt-3">
              <TrendChart
                type="line"
                data={dailyPoints}
                index="date"
                categories={["cache_read_input_tokens", "cache_creation_input_tokens"]}
                colors={[PALETTE.cacheRead, PALETTE.cacheWrite]}
                valueFormatter={formatCount}
              />
            </div>
          </SectionPanel>
        </div>

        <div className="mb-4">
          <SectionPanel title={`Spend by Model Over Time (top ${TOP_MODEL_SERIES})`}>
            <div className="mt-3">
              <TrendChart
                type="line"
                data={modelPoints}
                index="date"
                categories={modelSeries.models}
                colors={SERIES_COLORS.slice(0, modelSeries.models.length)}
                valueFormatter={formatSpend}
                height={288}
              />
            </div>
          </SectionPanel>
        </div>

        <div className="mb-4">
          <SectionPanel title="Top Models by Spend">
            <ShareBars rows={modelRows} labelOf={(row) => row.alias ?? row.id} />
          </SectionPanel>
        </div>
      </div>
    </>
  );
}
