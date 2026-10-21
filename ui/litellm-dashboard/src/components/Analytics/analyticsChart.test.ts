import { describe, it, expect } from "vitest";
import { buildChartOptions, buildChartSeries } from "./analyticsChart";

describe("buildChartSeries", () => {
  it("maps each category onto a named series in index order", () => {
    const data = [
      { date: "Oct 1", spend: 5, requests: 2 },
      { date: "Oct 2", spend: 7, requests: 0 },
    ];
    expect(buildChartSeries(data, ["spend", "requests"])).toEqual([
      { name: "spend", data: [5, 7] },
      { name: "requests", data: [2, 0] },
    ]);
  });

  it("zero-fills categories missing from a point", () => {
    expect(buildChartSeries([{ date: "Oct 1", spend: 3 }], ["spend", "requests"])).toEqual([
      { name: "spend", data: [3] },
      { name: "requests", data: [0] },
    ]);
  });
});

describe("buildChartOptions", () => {
  it("wires categories, colors, stacking, legend and formatting", () => {
    const options = buildChartOptions({
      labels: ["Oct 1", "Oct 2"],
      colors: ["#10b981", "#6366f1"],
      type: "bar",
      valueFormatter: (value: number) => `$${value}`,
      showLegend: false,
      stacked: true,
    });
    expect(options.chart).toMatchObject({ type: "bar", stacked: true });
    expect(options.colors).toEqual(["#10b981", "#6366f1"]);
    expect(options.xaxis.categories).toEqual(["Oct 1", "Oct 2"]);
    expect(options.legend.show).toBe(false);
    expect(options.tooltip.y.formatter(3)).toBe("$3");
    expect(options.yaxis.labels.formatter(4)).toBe("$4");
  });

  it("adds a gradient fill only for area charts", () => {
    const base = {
      labels: [],
      colors: [],
      valueFormatter: (value: number) => `${value}`,
      showLegend: true,
      stacked: false,
    };
    const area = buildChartOptions({ ...base, type: "area" });
    const line = buildChartOptions({ ...base, type: "line" });
    expect(area.fill).toMatchObject({ type: "gradient" });
    expect(line.fill).toBeUndefined();
  });
});
