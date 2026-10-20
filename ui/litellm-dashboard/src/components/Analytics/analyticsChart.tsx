"use client";

import dynamic from "next/dynamic";

const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

export type ChartKind = "area" | "line" | "bar";

export type ChartSeriesPoint = Record<string, string | number>;

export interface ChartSeries {
  name: string;
  data: number[];
}

export const buildChartSeries = (data: ChartSeriesPoint[], categories: string[]): ChartSeries[] =>
  categories.map((category) => ({
    name: category,
    data: data.map((point) => Number(point[category] ?? 0)),
  }));

export interface ChartOptionsInput {
  labels: string[];
  colors: string[];
  type: ChartKind;
  valueFormatter: (value: number) => string;
  showLegend: boolean;
  stacked: boolean;
}

export const buildChartOptions = ({
  labels,
  colors,
  type,
  valueFormatter,
  showLegend,
  stacked,
}: ChartOptionsInput) => ({
  chart: {
    type,
    stacked,
    toolbar: { show: false },
    fontFamily: "Inter, system-ui, sans-serif",
    background: "transparent",
    animations: { enabled: true, easing: "easeinout" as const, speed: 400 },
  },
  colors,
  dataLabels: { enabled: false },
  stroke: { curve: "smooth" as const, width: type === "bar" ? 1 : 2 },
  ...(type === "area"
    ? {
        fill: {
          type: "gradient" as const,
          gradient: { shadeIntensity: 1, opacityFrom: 0.7, opacityTo: 0.2, stops: [0, 90, 100] },
        },
      }
    : {}),
  ...(type === "bar" ? { plotOptions: { bar: { borderRadius: 2, columnWidth: "60%" } } } : {}),
  xaxis: {
    categories: labels,
    labels: {
      rotate: -45,
      trim: true,
      maxHeight: 80,
      style: { colors: "#6b7280", fontSize: "11px" },
    },
  },
  yaxis: { labels: { formatter: (value: number) => valueFormatter(value) } },
  tooltip: { shared: true, intersect: false, y: { formatter: (value: number) => valueFormatter(value) } },
  legend: { show: showLegend, position: "top" as const, horizontalAlign: "right" as const },
  grid: { borderColor: "#e5e7eb", strokeDashArray: 3 },
  theme: { mode: "light" as const },
});

export interface TrendChartProps {
  type: ChartKind;
  data: ChartSeriesPoint[];
  index: string;
  categories: string[];
  colors: string[];
  valueFormatter: (value: number) => string;
  showLegend?: boolean;
  stacked?: boolean;
  height?: number;
}

export const TrendChart = ({
  type,
  data,
  index,
  categories,
  colors,
  valueFormatter,
  showLegend = true,
  stacked = false,
  height = 224,
}: TrendChartProps) => {
  const labels = data.map((point) => String(point[index]));
  const series = buildChartSeries(data, categories);
  const options = buildChartOptions({ labels, colors, type, valueFormatter, showLegend, stacked });
  return <Chart options={options} series={series} type={type} height={height} />;
};
