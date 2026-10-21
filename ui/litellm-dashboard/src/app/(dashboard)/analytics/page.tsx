"use client";

import AnalyticsView from "@/components/Analytics/AnalyticsView";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function AnalyticsPage() {
  useAuthorized();
  return <AnalyticsView />;
}
