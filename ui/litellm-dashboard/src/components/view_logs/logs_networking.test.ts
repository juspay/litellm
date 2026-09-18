import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchUiSpendLogs } from "./logs_networking";

describe("fetchUiSpendLogs", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("requests fast pagination and forwards cancellation", async () => {
    const mockFetch = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        new Response(
          JSON.stringify({ data: [], total: null, page: 1, page_size: 50, total_pages: null, has_more: false }),
          { status: 200 },
        ),
      );
    global.fetch = mockFetch;
    const controller = new AbortController();

    await fetchUiSpendLogs({
      accessToken: "token",
      start_date: "2026-09-17 00:00:00",
      end_date: "2026-09-18 00:00:00",
      page: 1,
      page_size: 50,
      params: { request_id: "req-123" },
      signal: controller.signal,
    });

    const [url, options] = mockFetch.mock.calls[0];
    const parsed = new URL(String(url), "http://example.com");
    expect(parsed.pathname).toBe("/spend/logs/ui");
    expect(parsed.searchParams.get("include_total")).toBe("false");
    expect(parsed.searchParams.get("request_id")).toBe("req-123");
    expect(options?.signal).toBe(controller.signal);
  });
});
