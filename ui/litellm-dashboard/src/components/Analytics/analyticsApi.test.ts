import { beforeEach, describe, it, expect, vi } from "vitest";
import { keyListCall, teamListCall, userDailyActivityAggregatedCall } from "../networking";
import { DailyData } from "../UsagePage/types";
import { buildKeyInfo, buildTeamNames, loadMetadata, loadRecentDays, loadWindowDays, mergeDays } from "./analyticsApi";
import { RECENT_FETCH_DAYS } from "./analyticsConfig";
import { shiftDay, utcDayString } from "./analyticsData";

vi.mock("../networking", () => ({
  keyListCall: vi.fn(),
  teamListCall: vi.fn(),
  userDailyActivityAggregatedCall: vi.fn(),
}));

describe("buildKeyInfo", () => {
  it("maps rows by token with alias and budget", () => {
    const info = buildKeyInfo({
      keys: [
        { token: "hash-a", key_alias: "alpha", max_budget: 100 },
        { token: "hash-b", key_alias: null, max_budget: null },
      ],
    });

    expect(info.get("hash-a")).toEqual({ alias: "alpha", maxBudget: 100 });
    expect(info.get("hash-b")).toEqual({ alias: null, maxBudget: null });
    expect(info.size).toBe(2);
  });

  it("drops rows without a usable token", () => {
    const info = buildKeyInfo({
      keys: [{ key_alias: "no-token" }, { token: "", key_alias: "empty" }, { token: 7 }, null],
    });

    expect(info.size).toBe(0);
  });

  it("coerces non-string aliases and non-number budgets to null", () => {
    const info = buildKeyInfo({ keys: [{ token: "h", key_alias: 42, max_budget: "100" }] });

    expect(info.get("h")).toEqual({ alias: null, maxBudget: null });
  });

  it("returns an empty map for non-list payloads", () => {
    expect(buildKeyInfo(null).size).toBe(0);
    expect(buildKeyInfo({}).size).toBe(0);
    expect(buildKeyInfo({ keys: "nope" }).size).toBe(0);
  });
});

describe("buildTeamNames", () => {
  it("maps team ids to aliases and drops incomplete rows", () => {
    const names = buildTeamNames([
      { team_id: "t-1", team_alias: "platform" },
      { team_id: "t-2", team_alias: null },
      { team_alias: "orphan" },
      null,
    ]);

    expect(names.get("t-1")).toBe("platform");
    expect(names.size).toBe(1);
  });

  it("returns an empty map for non-array payloads", () => {
    expect(buildTeamNames({ teams: [] }).size).toBe(0);
    expect(buildTeamNames(null).size).toBe(0);
  });
});

describe("loadMetadata and loadWindowDays", () => {
  const fullPage = (offset: number, totalPages: number) => ({
    keys: Array.from({ length: 100 }, (_, index) => ({ token: `t${offset + index}` })),
    total_pages: totalPages,
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(userDailyActivityAggregatedCall).mockResolvedValue({ results: [] });
    vi.mocked(teamListCall).mockResolvedValue([]);
  });

  it("merges key info across every page until a short page", async () => {
    vi.mocked(keyListCall).mockImplementation((...args) =>
      Promise.resolve(
        args[6] === 1
          ? fullPage(0, 3)
          : args[6] === 2
            ? fullPage(100, 3)
            : { keys: [{ token: "t200", key_alias: "edge" }], total_pages: 3 },
      ),
    );

    const meta = await loadMetadata("token");

    expect(meta.keyInfo.size).toBe(201);
    expect(meta.keyInfo.get("t200")).toEqual({ alias: "edge", maxBudget: null });
    expect(keyListCall).toHaveBeenNthCalledWith(2, "token", null, null, null, null, null, 2, 100);
    expect(keyListCall).toHaveBeenCalledTimes(3);
  });

  it("stops at total_pages even when the last page is full", async () => {
    vi.mocked(keyListCall).mockResolvedValue(fullPage(0, 1));

    const meta = await loadMetadata("token");

    expect(meta.keyInfo.size).toBe(100);
    expect(keyListCall).toHaveBeenCalledTimes(1);
  });

  it("keeps earlier pages when a later page fails", async () => {
    vi.mocked(keyListCall).mockImplementation((...args) =>
      args[6] === 1 ? Promise.resolve(fullPage(0, 2)) : Promise.reject(new Error("page failed")),
    );

    const meta = await loadMetadata("token");

    expect(meta.keyInfo.size).toBe(100);
    expect(keyListCall).toHaveBeenCalledTimes(2);
  });

  it("degrades to empty key info when the first page fails", async () => {
    vi.mocked(keyListCall).mockRejectedValue(new Error("page failed"));

    const meta = await loadMetadata("token");

    expect(meta.keyInfo.size).toBe(0);
  });

  it("requests exactly the window plus its delta baseline as whole UTC days", async () => {
    await loadWindowDays("token", "5d");

    const [, start, end] = vi.mocked(userDailyActivityAggregatedCall).mock.calls[0];
    const today = utcDayString(new Date());
    expect(localDayLabel(end)).toBe(today);
    expect(localDayLabel(start)).toBe(shiftDay(today, -9));
  });
});

const localDayLabel = (date: Date): string =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;

describe("mergeDays", () => {
  it("replaces overlapping dates, appends new ones, and keeps ascending order", () => {
    const current = [{ date: "2026-10-18" }, { date: "2026-10-19" }] as DailyData[];
    const recent = [{ date: "2026-10-19" }, { date: "2026-10-20" }] as DailyData[];

    expect(mergeDays(current, recent).map((day) => day.date)).toEqual(["2026-10-18", "2026-10-19", "2026-10-20"]);
  });

  it("keeps the cached history untouched when nothing recent arrives", () => {
    const current = [{ date: "2026-10-18" }] as DailyData[];
    expect(mergeDays(current, [])).toEqual(current);
  });
});

describe("loadRecentDays", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("fetches only the recent window and never walks keys or teams", async () => {
    const recent = [{ date: "2026-10-19" }] as DailyData[];
    vi.mocked(userDailyActivityAggregatedCall).mockResolvedValue({ results: recent });

    const days = await loadRecentDays("token");

    expect(days).toEqual(recent);
    expect(keyListCall).not.toHaveBeenCalled();
    expect(teamListCall).not.toHaveBeenCalled();
    const [, start, end] = vi.mocked(userDailyActivityAggregatedCall).mock.calls[0];
    const today = utcDayString(new Date());
    expect(localDayLabel(end)).toBe(today);
    expect(localDayLabel(start)).toBe(shiftDay(today, -(RECENT_FETCH_DAYS - 1)));
  });

  it("returns no merge payload for malformed bodies", async () => {
    vi.mocked(userDailyActivityAggregatedCall).mockResolvedValue({});
    expect(await loadRecentDays("token")).toEqual([]);
  });
});
