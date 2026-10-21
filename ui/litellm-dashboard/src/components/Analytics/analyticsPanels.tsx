"use client";

import { ReactNode, useState } from "react";
import { TOP_N } from "./analyticsConfig";
import {
  BreakdownRow,
  HeatmapData,
  LabelRow,
  MetricKey,
  cacheHitRate,
  failRate,
  formatCount,
  formatMetric,
  formatSpend,
  metricValue,
  prettyDay,
  successRate,
} from "./analyticsData";

export const PANEL = "rounded-lg border border-gray-200 bg-white p-5 overflow-x-auto";
export const LABEL = "text-xs font-medium uppercase tracking-wider text-gray-500";
const CELL = "py-1.5 pr-2 text-right tabular-nums";
const BIG_NUMBER = "text-2xl sm:text-3xl font-bold tabular-nums text-gray-900 truncate";

export const rateText = (rate: number | null): string => (rate === null ? "-" : `${rate.toFixed(2)}%`);

export const rateColor = (rate: number | null): string => {
  if (rate === null) return "text-gray-400";
  if (rate >= 99) return "text-emerald-600";
  if (rate >= 95) return "text-amber-600";
  return "text-red-600";
};

const Empty = () => <div className="mt-3 text-sm text-gray-400">No data for this period</div>;

export const SectionPanel = ({ title, right, children }: { title: string; right?: ReactNode; children: ReactNode }) => (
  <div className={PANEL}>
    <div className="flex items-center justify-between gap-2">
      <h2 className={LABEL}>{title}</h2>
      {right}
    </div>
    {children}
  </div>
);

export const ToggleGroup = <T extends string>({
  options,
  value,
  onChange,
}: {
  options: readonly { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) => (
  <div className="flex gap-1">
    {options.map((option) => (
      <button
        key={option.value}
        type="button"
        onClick={() => onChange(option.value)}
        className={`rounded px-2 py-1 text-xs font-medium ${
          value === option.value ? "bg-emerald-600 text-white" : "bg-gray-100 text-gray-600 hover:bg-gray-200"
        }`}
      >
        {option.label}
      </button>
    ))}
  </div>
);

export const ToggleSwitch = ({
  checked,
  onChange,
  label,
  title,
  icon,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  title: string;
  icon: ReactNode;
}) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    title={title}
    aria-label={title}
    onClick={() => onChange(!checked)}
    className="flex items-center gap-2 px-1 text-xs font-medium text-gray-600 hover:text-gray-900"
  >
    {icon}
    {label}
    <span
      className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
        checked ? "bg-emerald-600" : "bg-gray-300"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
          checked ? "translate-x-4" : "translate-x-0.5"
        }`}
      />
    </span>
  </button>
);

export const DeltaBadge = ({ value }: { value: number | null }) =>
  value === null ? (
    <div className="mt-1 text-xs tabular-nums text-gray-400">no prior data</div>
  ) : (
    <div className="mt-1 text-xs tabular-nums text-gray-500">
      {value > 0 ? "+" : ""}
      {value.toFixed(1)}% vs prev
    </div>
  );

export const IconButton = ({
  title,
  onClick,
  spin = false,
  children,
}: {
  title: string;
  onClick: () => void;
  spin?: boolean;
  children: ReactNode;
}) => (
  <button
    type="button"
    title={title}
    aria-label={title}
    onClick={onClick}
    className="rounded bg-gray-100 p-1.5 text-base leading-none text-gray-600 hover:bg-gray-200"
  >
    <span className={spin ? "inline-flex animate-spin" : "inline-flex"}>{children}</span>
  </button>
);

export const Stat = ({
  label,
  value,
  sub,
  delta,
}: {
  label: string;
  value: string;
  sub?: string;
  delta?: number | null;
}) => (
  <div className={PANEL}>
    <div className={LABEL}>{label}</div>
    <div className={`${BIG_NUMBER} mt-2`}>{value}</div>
    {sub ? <div className="mt-1 text-sm tabular-nums text-gray-500">{sub}</div> : null}
    {delta !== undefined ? <DeltaBadge value={delta} /> : null}
  </div>
);

export const BudgetBar = ({ spend, maxBudget }: { spend: number; maxBudget: number | null }) => {
  if (maxBudget === null || maxBudget <= 0) return <span className="text-gray-400">-</span>;
  const ratio = Math.min(spend / maxBudget, 1);
  const color = ratio >= 0.9 ? "bg-red-500" : ratio >= 0.75 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="flex items-center justify-end gap-2">
      <div className="h-1.5 w-24 overflow-hidden rounded bg-gray-100">
        <div className={`h-full rounded ${color}`} style={{ width: `${ratio * 100}%` }} />
      </div>
      <span className="whitespace-nowrap text-xs tabular-nums text-gray-500">
        {Math.round(ratio * 100)}% of {formatSpend(maxBudget)}
      </span>
    </div>
  );
};

export const MetricsTable = ({
  rows,
  labelOf,
  extraHeader,
  extra,
}: {
  rows: BreakdownRow[];
  labelOf: (row: LabelRow) => string;
  extraHeader?: string;
  extra?: (row: BreakdownRow) => ReactNode;
}) => {
  const visible = rows.slice(0, TOP_N);
  return (
    <table className="mt-3 w-full text-sm tabular-nums">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="py-1.5 pr-2 font-medium">Name</th>
          <th className="py-1.5 pr-2 text-right font-medium">Spend</th>
          <th className="py-1.5 pr-2 text-right font-medium">Tokens</th>
          <th className="py-1.5 pr-2 text-right font-medium">Requests</th>
          <th className="py-1.5 text-right font-medium">OK%</th>
          {extraHeader ? <th className="py-1.5 pl-2 text-right font-medium">{extraHeader}</th> : null}
        </tr>
      </thead>
      <tbody>
        {visible.length === 0 ? (
          <tr>
            <td colSpan={extraHeader ? 6 : 5} className="py-3 text-center text-gray-400">
              No data for this period
            </td>
          </tr>
        ) : (
          visible.map((row) => (
            <tr key={row.id} className="border-t border-gray-200">
              <td className="max-w-[220px] truncate py-1.5 pr-2 text-gray-900" title={labelOf(row)}>
                {labelOf(row)}
              </td>
              <td className={`${CELL} text-gray-900`}>{formatSpend(row.metrics.spend)}</td>
              <td className={`${CELL} text-gray-500`}>{formatCount(row.metrics.total_tokens)}</td>
              <td className={`${CELL} text-gray-500`}>{formatCount(row.metrics.api_requests)}</td>
              <td className={`py-1.5 text-right ${rateColor(successRate(row.metrics))}`}>
                {rateText(successRate(row.metrics))}
              </td>
              {extra ? <td className={`${CELL} pl-2`}>{extra(row)}</td> : null}
            </tr>
          ))
        )}
      </tbody>
    </table>
  );
};

export const ShareBars = ({ rows, labelOf }: { rows: BreakdownRow[]; labelOf: (row: LabelRow) => string }) => {
  const total = rows.reduce((sum, row) => sum + row.metrics.spend, 0);
  if (rows.length === 0) return <Empty />;
  return (
    <div className="mt-3 space-y-2">
      {rows.slice(0, TOP_N).map((row) => {
        const share = total > 0 ? (row.metrics.spend / total) * 100 : 0;
        return (
          <div key={row.id}>
            <div className="flex justify-between gap-2 text-sm">
              <span className="truncate text-gray-900">{labelOf(row)}</span>
              <span className="whitespace-nowrap tabular-nums text-gray-500">
                {formatSpend(row.metrics.spend)} · {total > 0 ? `${share.toFixed(1)}%` : "-"}
              </span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded bg-gray-100">
              <div className="h-full rounded bg-emerald-500" style={{ width: `${share}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export const CacheTable = ({ rows, labelOf }: { rows: BreakdownRow[]; labelOf: (row: LabelRow) => string }) => {
  const sorted = [...rows]
    .filter((row) => row.metrics.cache_read_input_tokens + row.metrics.cache_creation_input_tokens > 0)
    .sort(
      (a, b) =>
        b.metrics.cache_read_input_tokens +
        b.metrics.cache_creation_input_tokens -
        (a.metrics.cache_read_input_tokens + a.metrics.cache_creation_input_tokens),
    )
    .slice(0, TOP_N);
  if (sorted.length === 0) return <Empty />;
  return (
    <table className="mt-3 w-full text-sm tabular-nums">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="py-1.5 pr-2 font-medium">Name</th>
          <th className="py-1.5 pr-2 text-right font-medium">Read</th>
          <th className="py-1.5 pr-2 text-right font-medium">Write</th>
          <th className="py-1.5 text-right font-medium">Hit%</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((row) => (
          <tr key={row.id} className="border-t border-gray-200">
            <td className="max-w-[220px] truncate py-1.5 pr-2 text-gray-900" title={labelOf(row)}>
              {labelOf(row)}
            </td>
            <td className={`${CELL} text-gray-500`}>{formatCount(row.metrics.cache_read_input_tokens)}</td>
            <td className={`${CELL} text-gray-500`}>{formatCount(row.metrics.cache_creation_input_tokens)}</td>
            <td className={`py-1.5 text-right ${rateColor(cacheHitRate(row.metrics))}`}>
              {rateText(cacheHitRate(row.metrics))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

type FailSort = "failRate" | "failed" | "requests";

export const FailureTable = ({ rows, labelOf }: { rows: BreakdownRow[]; labelOf: (row: LabelRow) => string }) => {
  const [sort, setSort] = useState<FailSort>("failRate");
  const sorted = [...rows]
    .filter((row) => row.metrics.api_requests > 0 && row.metrics.failed_requests > 0)
    .sort((a, b) => {
      if (sort === "failed") return b.metrics.failed_requests - a.metrics.failed_requests;
      if (sort === "requests") return b.metrics.api_requests - a.metrics.api_requests;
      return (failRate(b.metrics) ?? 0) - (failRate(a.metrics) ?? 0);
    })
    .slice(0, TOP_N);
  if (sorted.length === 0) return <Empty />;
  const header = (key: FailSort, label: string) => (
    <th className="py-1.5 pr-2 text-right font-medium">
      <button type="button" onClick={() => setSort(key)} className={sort === key ? "text-gray-900" : ""}>
        {label}
      </button>
    </th>
  );
  return (
    <table className="mt-3 w-full text-sm tabular-nums">
      <thead>
        <tr className="text-left text-gray-500">
          <th className="py-1.5 pr-2 font-medium">Name</th>
          {header("failed", "Failed")}
          {header("requests", "Requests")}
          {header("failRate", "Fail%")}
          <th className="py-1.5 text-right font-medium">OK%</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((row) => (
          <tr key={row.id} className="border-t border-gray-200">
            <td className="max-w-[220px] truncate py-1.5 pr-2 text-gray-900" title={labelOf(row)}>
              {labelOf(row)}
            </td>
            <td className={`${CELL} text-red-600`}>{formatCount(row.metrics.failed_requests)}</td>
            <td className={`${CELL} text-gray-500`}>{formatCount(row.metrics.api_requests)}</td>
            <td className={`${CELL} text-red-600`}>{rateText(failRate(row.metrics))}</td>
            <td className={`py-1.5 text-right ${rateColor(successRate(row.metrics))}`}>
              {rateText(successRate(row.metrics))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export const HeatmapTable = ({
  data,
  metric,
  labelOf,
  nameHeader = "Name",
  footerLabel,
}: {
  data: HeatmapData;
  metric: MetricKey;
  labelOf: (row: LabelRow) => string;
  nameHeader?: string;
  footerLabel: string;
}) => {
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full text-xs tabular-nums">
        <thead>
          <tr className="text-gray-500">
            <th className="py-1.5 pr-2 text-left font-medium">{nameHeader}</th>
            {data.dates.map((date) => (
              <th key={date} className="py-1.5 px-2 text-right font-medium">
                {prettyDay(date)}
              </th>
            ))}
            <th className="py-1.5 pl-2 text-right font-medium">Total</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.length === 0 ? (
            <tr>
              <td colSpan={data.dates.length + 2} className="py-3 text-center text-gray-400">
                No data for this period
              </td>
            </tr>
          ) : (
            data.rows.map((row) => (
              <tr key={row.id} className="border-t border-gray-200">
                <td className="max-w-[160px] truncate py-1.5 pr-2 text-gray-900" title={labelOf(row)}>
                  {labelOf(row)}
                </td>
                {row.cells.map((cell, index) => (
                  <td key={data.dates[index]} className="px-2 py-1.5 text-right text-gray-700">
                    {metricValue(cell, metric) === 0 ? "-" : formatMetric(cell, metric)}
                  </td>
                ))}
                <td className="py-1.5 pl-2 text-right font-medium text-gray-900">
                  {formatMetric(row.metrics, metric)}
                </td>
              </tr>
            ))
          )}
          {data.rows.length > 0 ? (
            <tr className="border-t border-gray-300 font-semibold">
              <td className="py-1.5 pr-2 text-gray-700">{footerLabel}</td>
              {data.columnTotals.map((value, index) => (
                <td key={data.dates[index]} className="px-2 py-1.5 text-right text-gray-900">
                  {formatMetric(value, metric)}
                </td>
              ))}
              <td className="py-1.5 pl-2 text-right font-medium text-gray-900">{formatMetric(data.totals, metric)}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
};

const SHIMMER_KEYFRAMES = "@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}";

export const Skeleton = ({ className = "" }: { className?: string }) => (
  <div className={`relative overflow-hidden rounded bg-gray-200 ${className}`}>
    <div
      className="absolute inset-0 bg-gradient-to-r from-transparent via-white/80 to-transparent"
      style={{ animation: "shimmer 1.6s linear infinite" }}
    />
  </div>
);

export const SkeletonStatCard = () => (
  <div className={PANEL}>
    <Skeleton className="h-3 w-24" />
    <Skeleton className="mt-3 h-8 w-32" />
    <Skeleton className="mt-3 h-3 w-40" />
    <Skeleton className="mt-2 h-3 w-28" />
  </div>
);

export const SkeletonTable = ({ rows = 8 }: { rows?: number }) => (
  <div className={PANEL}>
    <Skeleton className="h-3 w-40" />
    <div className="mt-3 space-y-2">
      {Array.from({ length: rows + 1 }, (_, index) => (
        <Skeleton key={index} className="h-3 w-full" />
      ))}
    </div>
  </div>
);

export const SkeletonChart = () => (
  <div className={PANEL}>
    <Skeleton className="h-3 w-40" />
    <Skeleton className="mt-3 h-56 w-full" />
  </div>
);

export const SkeletonHeatmap = () => (
  <div className={PANEL}>
    <Skeleton className="h-3 w-48" />
    <div className="mt-3 space-y-2">
      {Array.from({ length: 6 }, (_, row) => (
        <div key={row} className="flex gap-2">
          <Skeleton className="h-4 w-32" />
          {Array.from({ length: 7 }, (_, cell) => (
            <Skeleton key={cell} className="h-4 flex-1" />
          ))}
        </div>
      ))}
    </div>
  </div>
);

export const AnalyticsSkeleton = ({ variant = "tables" }: { variant?: "tables" | "graphs" }) => (
  <div aria-busy="true" aria-label="Loading usage data">
    <style>{SHIMMER_KEYFRAMES}</style>
    <div className="mb-4 grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
      {Array.from({ length: 6 }, (_, index) => (
        <SkeletonStatCard key={index} />
      ))}
    </div>
    {variant === "graphs" ? (
      <>
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SkeletonChart />
          <SkeletonChart />
        </div>
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SkeletonChart />
          <SkeletonChart />
        </div>
        <div className="mb-4">
          <SkeletonChart />
        </div>
        <div className="mb-4">
          <SkeletonTable />
        </div>
      </>
    ) : (
      <>
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SkeletonTable />
          <SkeletonHeatmap />
        </div>
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SkeletonTable />
          <SkeletonTable />
        </div>
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SkeletonTable />
          <SkeletonChart />
        </div>
        <div className="mb-4 grid gap-4 items-start xl:grid-cols-2">
          <SkeletonTable rows={5} />
          <SkeletonTable rows={5} />
        </div>
      </>
    )}
  </div>
);
