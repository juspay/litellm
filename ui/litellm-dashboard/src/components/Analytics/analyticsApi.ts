import { keyListCall, teamListCall, userDailyActivityAggregatedCall } from "../networking";
import { DailyData } from "../UsagePage/types";
import { RECENT_FETCH_DAYS } from "./analyticsConfig";
import { AnalyticsWindow, shiftDay, utcDayString, windowFetchRange } from "./analyticsData";

export interface KeyInfo {
  alias: string | null;
  maxBudget: number | null;
}

export type KeyInfoMap = Map<string, KeyInfo>;

export interface AnalyticsBundle {
  days: DailyData[];
  keyInfo: KeyInfoMap;
  teamNames: Map<string, string>;
  updatedAt: Date | null;
}

export const emptyBundle = (): AnalyticsBundle => ({
  days: [],
  keyInfo: new Map(),
  teamNames: new Map(),
  updatedAt: null,
});

const asRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};

const asString = (value: unknown): string | null => (typeof value === "string" && value.length > 0 ? value : null);

const asBudget = (value: unknown): number | null => (typeof value === "number" ? value : null);

export const buildKeyInfo = (body: unknown): KeyInfoMap => {
  const rows = asRecord(body).keys;
  if (!Array.isArray(rows)) return new Map();
  return new Map(
    rows.flatMap((entry): Array<readonly [string, KeyInfo]> => {
      const row = asRecord(entry);
      const token = asString(row.token);
      return token === null ? [] : [[token, { alias: asString(row.key_alias), maxBudget: asBudget(row.max_budget) }]];
    }),
  );
};

export const buildTeamNames = (body: unknown): Map<string, string> => {
  if (!Array.isArray(body)) return new Map();
  return new Map(
    body.flatMap((entry): Array<readonly [string, string]> => {
      const row = asRecord(entry);
      const id = asString(row.team_id);
      const alias = asString(row.team_alias);
      return id === null || alias === null ? [] : [[id, alias]];
    }),
  );
};

const KEY_PAGE_SIZE = 100;

const dayBoundary = (label: string): Date => {
  const [year, month, day] = label.split("-").map(Number);
  return new Date(year, month - 1, day);
};

const toDates = (range: readonly [string, string]): readonly [Date, Date] => [
  dayBoundary(range[0]),
  dayBoundary(range[1]),
];

const recentRange = (): readonly [string, string] => {
  const today = utcDayString(new Date());
  return [shiftDay(today, -(RECENT_FETCH_DAYS - 1)), today];
};

export const mergeDays = (current: DailyData[], recent: DailyData[]): DailyData[] => {
  const freshDates = new Set(recent.map((day) => day.date));
  return [...current.filter((day) => !freshDates.has(day.date)), ...recent].sort((a, b) =>
    a.date.localeCompare(b.date),
  );
};

const fetchUsage = async (accessToken: string, range: readonly [string, string]): Promise<DailyData[]> => {
  const [start, end] = toDates(range);
  const usage = await userDailyActivityAggregatedCall(accessToken, start, end, null);
  const results = asRecord(usage).results;
  return Array.isArray(results) ? (results as DailyData[]) : [];
};

export const loadWindowDays = (accessToken: string, win: AnalyticsWindow): Promise<DailyData[]> =>
  fetchUsage(accessToken, windowFetchRange(win));

export const loadRecentDays = (accessToken: string): Promise<DailyData[]> => fetchUsage(accessToken, recentRange());

const fetchKeyInfo = async (accessToken: string, page = 1, acc: KeyInfoMap = new Map()): Promise<KeyInfoMap> => {
  const body = await keyListCall(accessToken, null, null, null, null, null, page, KEY_PAGE_SIZE).catch(() => null);
  const merged = new Map([...acc, ...buildKeyInfo(body)]);
  const rows = asRecord(body).keys;
  const totalPages = asRecord(body).total_pages;
  const hasMore =
    Array.isArray(rows) && rows.length === KEY_PAGE_SIZE && (typeof totalPages !== "number" || page < totalPages);
  return hasMore ? fetchKeyInfo(accessToken, page + 1, merged) : merged;
};

export const loadMetadata = async (accessToken: string): Promise<Pick<AnalyticsBundle, "keyInfo" | "teamNames">> => {
  const [teams, keyInfo] = await Promise.all([
    teamListCall(accessToken, null).catch(() => null),
    fetchKeyInfo(accessToken),
  ]);
  return { keyInfo, teamNames: buildTeamNames(teams) };
};
