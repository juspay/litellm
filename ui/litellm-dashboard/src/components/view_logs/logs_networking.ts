import { createApiClient } from "@/lib/http/client";
import { getGlobalLitellmHeaderName, getProxyBaseUrl, handleError } from "../networking";
import type { PaginatedResponse } from "./log_filter_logic";

interface UiSpendLogsParams {
  api_key?: string;
  team_id?: string;
  request_id?: string;
  user_id?: string;
  end_user?: string;
  status_filter?: string;
  model?: string;
  model_id?: string;
  key_alias?: string;
  error_code?: string;
  error_message?: string;
  sort_by?: string;
  sort_order?: "asc" | "desc";
}

interface UiSpendLogsCallOptions {
  accessToken: string;
  start_date: string;
  end_date: string;
  page: number;
  page_size: number;
  params: UiSpendLogsParams;
  signal?: AbortSignal;
}

const logsApiClient = createApiClient({
  getBaseUrl: getProxyBaseUrl,
  getAuthHeaderName: getGlobalLitellmHeaderName,
  onError: handleError,
});

export const fetchUiSpendLogs = async ({
  accessToken,
  start_date,
  end_date,
  page,
  page_size,
  params,
  signal,
}: UiSpendLogsCallOptions): Promise<PaginatedResponse> => {
  const optionalQuery = Object.fromEntries(
    Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== ""),
  );
  return logsApiClient.get<PaginatedResponse>("/spend/logs/ui", {
    accessToken,
    signal,
    query: {
      start_date,
      end_date,
      page,
      page_size,
      include_total: false,
      ...optionalQuery,
    },
  });
};
