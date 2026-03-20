-- CreateTable
CREATE TABLE "LiteLLM_AggregatedSpendLogs" (
    "id" SERIAL PRIMARY KEY,

    -- Time dimension
    "spend_date" DATE NOT NULL,

    -- Entity dimensions
    "user_id" TEXT NOT NULL,
    "model" TEXT NOT NULL DEFAULT '',
    "custom_llm_provider" TEXT NOT NULL DEFAULT '',

    -- Aggregated metrics
    "total_spend" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "total_tokens" BIGINT NOT NULL DEFAULT 0,
    "prompt_tokens" BIGINT NOT NULL DEFAULT 0,
    "completion_tokens" BIGINT NOT NULL DEFAULT 0,
    "total_requests" BIGINT NOT NULL DEFAULT 0,
    "total_duration_ms" BIGINT NOT NULL DEFAULT 0,

    -- Metadata
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    -- Constraint
    CONSTRAINT "LiteLLM_AggregatedSpendLogs_spend_date_user_id_model_custom_llm_provider_key" UNIQUE ("spend_date", "user_id", "model", "custom_llm_provider")
);

-- CreateIndex (spend_date is covered by the unique constraint's leftmost prefix)
CREATE INDEX "LiteLLM_AggregatedSpendLogs_user_id_idx" ON "LiteLLM_AggregatedSpendLogs"("user_id");

-- CreateIndex
CREATE INDEX "LiteLLM_AggregatedSpendLogs_model_idx" ON "LiteLLM_AggregatedSpendLogs"("model");

-- CreateIndex
CREATE INDEX "LiteLLM_AggregatedSpendLogs_custom_llm_provider_idx" ON "LiteLLM_AggregatedSpendLogs"("custom_llm_provider");

-- CreateTrigger: Auto-update updated_at on row modification (table-specific function)
CREATE OR REPLACE FUNCTION update_aggregated_spend_logs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS trigger_updated_at_aggregated_spend_logs ON "LiteLLM_AggregatedSpendLogs";
CREATE TRIGGER trigger_updated_at_aggregated_spend_logs
    BEFORE UPDATE ON "LiteLLM_AggregatedSpendLogs"
    FOR EACH ROW
    EXECUTE FUNCTION update_aggregated_spend_logs_updated_at();
