# Provider prompt-cache metrics: findings and implementation notes

Working notes behind the `cache_read_input_tokens` column on `LiteLLM_SpendLogs`. The goal was to record, per request, how much of the prompt the inference server served from its KV cache instead of recomputing. Everything below was measured against live vLLM and sglang deployments rather than taken from documentation

## 1. The servers already report it

Both engines return the value on every request, under the OpenAI-compatible key `usage.prompt_tokens_details.cached_tokens`. No client-side estimation is needed and none should be written

Measured behaviour, sending the same long prompt twice:

| Engine | Cold call | Warm call |
| --- | --- | --- |
| vLLM (glm) | `cached_tokens: 0` | 5632 of 6431 |
| vLLM (kimi-k2) | `cached_tokens: 512` | 6912 of 6969 |
| vLLM (deepseek) | `cached_tokens: 0` | 6144 of 6423 |
| sglang (kimi-k3) | `prompt_tokens_details: null` | 6400 of 6514 |

Note the shape difference on a cold call: vLLM sends `{"cached_tokens": 0}`, sglang omits the object entirely. Any code doing `usage.prompt_tokens_details.cached_tokens` without a guard will fail on sglang

LiteLLM passes the field through unchanged on both the `hosted_vllm/` and `litellm_proxy/` provider paths, streaming and non-streaming

## 2. It is ground truth, not an estimate

vLLM stores KV blocks in a dict keyed by a chained hash of the block's tokens (`cached_block_hash_to_block` in `vllm/v1/core/block_pool.py`). Each block's hash folds in its parent's, so one lookup answers "do I hold exactly this N-token prefix". A hit means the scheduler hands the request existing GPU tensors and skips prefill

So `cached_tokens` reports an allocation decision that already happened. It is not the server guessing

## 3. The number is a floor, never exact

Three effects make the reported value lower than true reuse

**Only full blocks count.** The trailing partial block is never hashed or stored, so `prompt_tokens % granularity` tokens are permanently ineligible

**A block is lost when the prompt is fully cached.** From `vllm/v1/core/kv_cache_manager.py`:

```python
# NOTE: When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
# This can trigger recomputation of an entire block, rather than just
# the single last token, because allocate_slots() requires
# num_computed_tokens to be block-size aligned.
max_cache_hit_length = request.num_tokens - 1
```

A forward pass needs at least one token to produce logits, and allocation must be block-aligned, so holding back one token costs a whole block. Demonstrated with a prompt built to be exactly 768 tokens on a 128-granularity deployment: every block was cached, yet it reported 640 (83.3%). An A/B test confirmed the cause; resending an identical prompt lost a block, while the same prefix followed by a new tail lost nothing

**Granularity varies per deployment and is not derivable from `block_size`.** Measured:

| Deployment | Configured `block_size` | Observed granularity | Smallest prompt that can report a hit |
| --- | --- | --- | --- |
| glm (MLA, DCP=8) | 64 (auto) | 512 | 1024 tok |
| kimi-k2 | 16 (auto) | 128 | 129 tok |
| deepseek (`--block-size 4`) | 4 (explicit) | 1024 | 1025 tok |
| kimi-k3 (sglang, `page_size` 64) | 64 | 256 | 256 tok |

On the glm deployment the effective granularity is `block_size * decode_context_parallel_size` (64 * 8), since DCP shards the cache across ranks and a prefix is only reusable when aligned across all of them. The other deployments do not follow that rule, so granularity must be measured per endpoint

This is why a short prompt always reports zero. A 19-token request cannot fill a single block on any of these deployments

## 4. The two engines report differently

vLLM applies the `-1` rule to the reported number; sglang does not. Same situation, block-aligned prompt with every token cached:

| Engine | prompt_tokens | cached_tokens | Reported ratio |
| --- | --- | --- | --- |
| vLLM | 768 | 640 | 83.3% |
| sglang | 512 | 512 | 100.0% |

Cache hit percentages are therefore not comparable across engines. An sglang deployment will look better than a vLLM one on identical traffic. Store enough identity on each row (`model_id` or `api_base`) to segment by engine before comparing anything

## 5. Cache read and cache write are different things

Cache read is prefix reuse. Cache write is tokens billed for being stored, which only exists where caching is explicit and charged, as on Anthropic

vLLM and sglang have no write concept. Prefix caching is automatic and free, so there is no event to report and no column to fill. A cache-write field would be null on every row these deployments produce, which is why only `cache_read_input_tokens` was added

The conventions also differ in a way that breaks naive arithmetic. On OpenAI-compatible usage `prompt_tokens` already includes `cached_tokens`, so the uncached work is `prompt_tokens - cached_tokens`. On Anthropic, `input_tokens` excludes the cache fields and they must be added. LiteLLM normalises this before costing in `litellm/cost_calculator.py`

## 6. Instrumentation gaps worth auditing

`--enable-prompt-tokens-details` is required on vLLM. It defaults to `False`, and without it `_make_prompt_tokens_details` returns `None` and the whole object is omitted. Two deployments in the fleet return no `prompt_tokens_details` at all despite `enable_prefix_caching="True"` and server-side counters showing 64.9% and 79.5% real reuse. Their cache works fine; only the per-request field is missing

Both silent deployments are hybrid Mamba models running two KV cache groups, so the cause is either the missing flag or a `None` `num_cached_tokens` for hybrid models. The two are indistinguishable over HTTP; confirm from the `non-default args:` line in the server's startup log

Streaming requests need `stream_options: {"include_usage": true}`. Without it there is no usage block at all, so nothing is logged and the row reads as zero cache when the truth is unknown

## 7. Server-side counters are the calibration source

vLLM exposes exact aggregates on `/metrics`, free of the block-granularity distortion:

```
vllm:prompt_tokens_by_source_total{source="local_compute"}          10.0%
vllm:prompt_tokens_by_source_total{source="local_cache_hit"}        72.1%
vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}   17.8%
vllm:prompt_tokens_cached_total                                     90.0% of prompt tokens
```

Only 10% of prompt tokens on that deployment are actually computed. Use these counters to sanity-check whatever the per-request rows aggregate to; a gap between them is the measurement floor, not a bug. What they cannot do is attribute reuse to a key, team or session, which is the reason the per-request column exists

`--kv-cache-metrics-sample` affects only three histograms (`kv_block_lifetime_seconds`, `kv_block_idle_before_evict_seconds`, `kv_block_reuse_gap_seconds`). Counters and the per-request field are unsampled

## 8. What was implemented

`litellm/proxy/spend_tracking/cache_tokens.py` holds a single provider-agnostic extractor returning `Optional[int]`, used by both the spend log and the daily rollup writer so they cannot drift. It handles the Anthropic top-level fields, the OpenAI-compatible `prompt_tokens_details` shape, sglang's null object, and vLLM's `created_cache_tokens` spelling, which the previous extractor did not check

`cache_read_input_tokens Int?` was added to `LiteLLM_SpendLogs` and is populated in `get_logging_payload`

The column is nullable on purpose. `NULL` means the provider reported nothing, `0` means it reported a genuine miss. Collapsing them would make the un-instrumented deployments in section 6 look like total cache failures and would quietly bias every aggregate built on the column

The dashboard log drawer reads the column and shows Cache Read Tokens plus a derived Cache Hit Ratio, falling back to the metadata JSON so rows written before the change still render. The pre-existing Cache Hit row was left alone; it reflects LiteLLM's own response cache, which is unrelated to provider-side prompt caching and is a common source of confusion when reading these logs

## 9. No ratio is stored

The ratio is derived at read time as `SUM(cache_read_input_tokens) / SUM(prompt_tokens)`

Storing a per-row percentage would be worse in three ways. Percentages cannot be re-aggregated, so averaging them weights a 20-token request the same as a 200k one. The value is redundant, since `prompt_tokens` is already on the row. And it forces a number onto rows whose real answer is "not measured"

When computing a fleet number, prefer the sum-of-counts form over an average of per-request ratios, and segment by endpoint so the differing granularities and engine conventions do not get mixed together

## 10. Still outstanding

Tests have not been written or run for the new column. `prisma generate` and a schema push are required before anything populates. `npm run gen:api` has not been run, which CI enforces. The dashboard changes have not been typechecked
