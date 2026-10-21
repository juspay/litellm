"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { DailyData } from "../UsagePage/types";
import { AnalyticsBundle, emptyBundle, loadMetadata, loadRecentDays, loadWindowDays, mergeDays } from "./analyticsApi";
import { METADATA_REFRESH_MS } from "./analyticsConfig";
import { AnalyticsWindow, coversRange, windowFetchRange } from "./analyticsData";

type DaysUpdate = (prev: AnalyticsBundle) => AnalyticsBundle;

const mergeRecent =
  (recent: DailyData[]): DaysUpdate =>
  (prev) => ({
    ...prev,
    days: mergeDays(prev.days, recent),
    updatedAt: new Date(),
  });

export const useAnalyticsData = (
  accessToken: string | null,
  windowKey: AnalyticsWindow,
  autoRefreshMs: number | null,
) => {
  const [bundle, setBundle] = useState<AnalyticsBundle>(emptyBundle);
  const [loading, setLoading] = useState(accessToken !== null);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);
  const fetchedRange = useRef<readonly [string, string] | null>(null);
  const metadataAge = useRef(0);

  const settle = useCallback((promise: Promise<DaysUpdate>) => {
    const id = requestId.current + 1;
    requestId.current = id;
    promise
      .then((apply) => {
        if (requestId.current !== id) return;
        setBundle((prev) => apply(prev));
        setError(null);
      })
      .catch(() => {
        if (requestId.current !== id) return;
        setError("Failed to load usage data");
      })
      .finally(() => {
        if (requestId.current === id) setLoading(false);
      });
  }, []);

  const fetchWindow = useCallback(
    (force: boolean) => {
      if (!accessToken) return;
      const range = windowFetchRange(windowKey);
      if (!force && coversRange(fetchedRange.current, range)) return;
      settle(
        loadWindowDays(accessToken, windowKey).then((days) => {
          fetchedRange.current = range;
          return (prev: AnalyticsBundle): AnalyticsBundle => ({ ...prev, days, updatedAt: new Date() });
        }),
      );
    },
    [accessToken, settle, windowKey],
  );

  const fetchMetadata = useCallback(() => {
    if (!accessToken) return;
    metadataAge.current = Date.now();
    loadMetadata(accessToken)
      .then((meta) => setBundle((prev) => ({ ...prev, ...meta })))
      .catch(() => undefined);
  }, [accessToken]);

  useEffect(() => {
    fetchWindow(false);
  }, [fetchWindow]);

  useEffect(() => {
    fetchMetadata();
  }, [fetchMetadata]);

  const tick = useCallback(() => {
    if (!accessToken) return;
    if (Date.now() - metadataAge.current >= METADATA_REFRESH_MS) fetchMetadata();
    settle(loadRecentDays(accessToken).then(mergeRecent));
  }, [accessToken, fetchMetadata, settle]);

  useEffect(() => {
    if (!autoRefreshMs) return;
    const timer = setInterval(tick, autoRefreshMs);
    return () => clearInterval(timer);
  }, [autoRefreshMs, tick]);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchWindow(true);
    fetchMetadata();
  }, [fetchWindow, fetchMetadata]);

  return { ...bundle, loading, error, refresh };
};
