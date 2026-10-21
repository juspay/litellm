import { renderHook, waitFor, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DailyData } from "../UsagePage/types";
import { loadMetadata, loadRecentDays, loadWindowDays } from "./analyticsApi";
import { METADATA_REFRESH_MS } from "./analyticsConfig";
import { AnalyticsWindow } from "./analyticsData";
import { useAnalyticsData } from "./useAnalyticsData";

vi.mock("./analyticsApi", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./analyticsApi")>();
  return { ...actual, loadWindowDays: vi.fn(), loadMetadata: vi.fn(), loadRecentDays: vi.fn() };
});

const mockedDays = loadWindowDays as unknown as ReturnType<typeof vi.fn>;
const mockedMeta = loadMetadata as unknown as ReturnType<typeof vi.fn>;
const mockedRecent = loadRecentDays as unknown as ReturnType<typeof vi.fn>;

const day = (date: string): DailyData => ({ date }) as DailyData;

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

beforeEach(() => {
  mockedDays.mockReset();
  mockedMeta.mockReset();
  mockedRecent.mockReset();
  mockedMeta.mockResolvedValue({ keyInfo: new Map(), teamNames: new Map() });
});

describe("useAnalyticsData", () => {
  it("loads the window on mount, ending the skeleton with the fetched days", async () => {
    mockedDays.mockResolvedValue([day("2026-10-20")]);

    const { result } = renderHook(() => useAnalyticsData("token", "today", null));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBeNull();
    expect(result.current.days.map((entry) => entry.date)).toEqual(["2026-10-20"]);
    expect(result.current.updatedAt).toBeInstanceOf(Date);
    expect(mockedDays).toHaveBeenCalledWith("token", "today");
    expect(mockedMeta).toHaveBeenCalledTimes(1);
  });

  it("does not fetch and clears the skeleton without an access token", async () => {
    const { result } = renderHook(() => useAnalyticsData(null, "today", null));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(mockedDays).not.toHaveBeenCalled();
    expect(mockedMeta).not.toHaveBeenCalled();
  });

  it("manual refresh shows the skeleton and swaps in fresh days and metadata", async () => {
    mockedDays.mockResolvedValueOnce([day("2026-10-20")]);
    const { result } = renderHook(() => useAnalyticsData("token", "today", null));
    await waitFor(() => expect(result.current.loading).toBe(false));

    const slow = deferred<DailyData[]>();
    mockedDays.mockReturnValueOnce(slow.promise);
    act(() => {
      result.current.refresh();
    });
    expect(result.current.loading).toBe(true);

    await act(async () => {
      slow.resolve([day("2026-10-21")]);
      await slow.promise;
    });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.days.map((entry) => entry.date)).toEqual(["2026-10-21"]);
    expect(mockedDays).toHaveBeenCalledTimes(2);
    expect(mockedMeta).toHaveBeenCalledTimes(2);
  });

  it("refetches only the days when the window widens and re-slices instantly when covered", async () => {
    mockedDays.mockResolvedValue([day("2026-10-11"), day("2026-10-20")]);
    const { result, rerender } = renderHook(
      ({ win }: { win: AnalyticsWindow }) => useAnalyticsData("token", win, null),
      {
        initialProps: { win: "today" as AnalyticsWindow },
      },
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(mockedDays).toHaveBeenCalledTimes(1);

    rerender({ win: "30d" });
    await waitFor(() => expect(mockedDays).toHaveBeenCalledTimes(2));
    expect(mockedDays).toHaveBeenLastCalledWith("token", "30d");

    rerender({ win: "5d" });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(mockedDays).toHaveBeenCalledTimes(2);
    expect(mockedMeta).toHaveBeenCalledTimes(1);
  });

  it("ignores a stale days response that settles after a newer request", async () => {
    const stale = deferred<DailyData[]>();
    const fresh = deferred<DailyData[]>();
    mockedDays.mockReturnValueOnce(stale.promise).mockReturnValueOnce(fresh.promise);

    const { result } = renderHook(() => useAnalyticsData("token", "today", null));
    act(() => {
      result.current.refresh();
    });

    await act(async () => {
      fresh.resolve([day("2026-10-20")]);
      await fresh.promise;
    });
    await act(async () => {
      stale.resolve([day("2026-10-01")]);
      await stale.promise;
    });

    await waitFor(() => expect(result.current.days.map((entry) => entry.date)).toEqual(["2026-10-20"]));
    expect(result.current.loading).toBe(false);
  });

  it("surfaces a load failure and recovers on the next refresh", async () => {
    mockedDays.mockRejectedValueOnce(new Error("boom"));
    const { result } = renderHook(() => useAnalyticsData("token", "today", null));

    await waitFor(() => expect(result.current.error).toBe("Failed to load usage data"));
    expect(result.current.loading).toBe(false);

    mockedDays.mockResolvedValueOnce([day("2026-10-20")]);
    await act(async () => {
      result.current.refresh();
    });
    await waitFor(() => expect(result.current.days.map((entry) => entry.date)).toEqual(["2026-10-20"]));
    expect(result.current.error).toBeNull();
  });

  it("refreshes silently on the interval without re-showing the skeleton", async () => {
    vi.useFakeTimers();
    try {
      mockedDays.mockResolvedValue([day("2026-10-20")]);
      mockedRecent.mockResolvedValue([]);
      const { result } = renderHook(() => useAnalyticsData("token", "today", 1000));

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(result.current.loading).toBe(false);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
      expect(mockedDays).toHaveBeenCalledTimes(1);
      expect(mockedRecent).toHaveBeenCalledTimes(1);
      expect(result.current.loading).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("merges recent days on silent ticks without refetching the window", async () => {
    vi.useFakeTimers();
    try {
      mockedDays.mockResolvedValue([day("2026-10-19")]);
      mockedRecent.mockResolvedValue([day("2026-10-20")]);
      const { result } = renderHook(() => useAnalyticsData("token", "today", 1000));

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });

      expect(mockedDays).toHaveBeenCalledTimes(1);
      expect(mockedRecent).toHaveBeenCalledWith("token");
      expect(result.current.days.map((entry) => entry.date)).toEqual(["2026-10-19", "2026-10-20"]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("refreshes key metadata on a tick once it goes stale", async () => {
    vi.useFakeTimers();
    try {
      mockedDays.mockResolvedValue([day("2026-10-20")]);
      mockedRecent.mockResolvedValue([]);
      const { result } = renderHook(() => useAnalyticsData("token", "today", METADATA_REFRESH_MS));

      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(METADATA_REFRESH_MS);
      });

      expect(mockedMeta).toHaveBeenCalledTimes(2);
      expect(mockedRecent).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });
});
