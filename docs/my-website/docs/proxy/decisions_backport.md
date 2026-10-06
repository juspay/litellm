# Decisions API on v1.92.0

This backports BerriAI/LiteLLM PR #44236 (commit `8b1990b4bcc61a98da08bb71e7519d6b53c8572a`) and adds optional top-level `images` forwarding to synchronous and asynchronous Decisions calls.

## Configure a self-hosted SystemOne model

The upstream `strands_decider` adapter implements the SystemOne protocol and can route to a compatible self-hosted wrapper. The client-facing alias is independent of the upstream model name.

```yaml
model_list:
  - model_name: xor
    litellm_params:
      model: strands_decider/jev-trained
      api_base: os.environ/XOR_API_BASE
    model_info:
      mode: evaluation

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Set `XOR_API_BASE` to the wrapper's origin, such as `http://localhost:8000`, or its `/v1` base. The adapter appends `/v1/systemone`. Set `litellm_params.api_key` if the wrapper requires authentication. Configure input/output pricing explicitly if infrastructure cost should appear in spend accounting; the provider name does not supply a price for a custom model.

For an extraction deployment, `api_base` can be the complete `/v1/generate` URL. Complete `/v1/systemone` URLs are also accepted without appending the path again. Keep classification and extraction deployments as separate aliases when their upstream endpoints differ.

Grant the team's and virtual key's normal model permissions access to `xor`. Multiple deployments with `model_name: xor` use the normal Router selection and fallback pipeline.

## Call Decisions

```bash
curl -sS --fail-with-body "$LITELLM_BASE_URL/v1/decisions" \
  -H "Authorization: Bearer $JUSPAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "xor",
    "state": "The customer was charged twice.",
    "questions": {
      "billing": {"type": "noul", "instructions": "Is this a billing issue?"}
    }
  }'
```

For images, add `"images": ["data:image/png;base64,..."]` or image URLs accepted by the wrapper. LiteLLM forwards the strings without downloading or transforming them. The selected upstream must support the supplied format. Text-only calls omit `images` from the upstream body.

## Audio and extraction

The self-hosted wrapper extensions accept `"audio": {"data": "<base64>", "format": "wav"}`. LiteLLM forwards the audio without decoding or transcoding; the upstream determines supported encodings. Supply `state`, `context`, or audio as grounding. If both `state` and `context` are supplied, both are forwarded unchanged.

For free-form extraction, use `"questions": {"author": {"type": "string", "instructions": "Who is the author?"}}` with `"context": "This paper was written by Example Author."`. The response preserves the upstream extraction envelope: `result`, `usage`, `thinking`, and per-field `confidence` containing `mean_p` and `min_p`. It does not convert this envelope into classification `answers`.

The alternative `value` question used by voice clients accepts an optional `pattern` and returns an `answers` entry with `type: value`, an extracted JSON `value`, and optional `confidence` and unknown/abstention fields. These are separate wire formats rather than aliases.

Extraction usage reports visible `completion_tokens` and separate `thinking_tokens`. Their sum is normalized internally as output tokens for cost, logging and TPM accounting, with thinking recorded as reasoning-token details. The original usage fields remain unchanged in the public response. Upstreams must not include thinking tokens again in `completion_tokens`.

Audio and extraction require a capable upstream; this extension does not add those capabilities to third-party providers. Mixed extraction/classification inference is not verified and remains upstream-dependent. Legacy pass-through routes are unaffected.

Decisions-aware unified guardrails scan the JSON containing `state`, `context`, and `questions`, including instructions and criteria. Sanitized JSON is validated before forwarding. Image references are supplied to image-aware guardrails. Extraction and classification output guardrails scan the response JSON while preserving original usage and billing metadata. Legacy message-based input guardrails have their sanitized JSON written back into the Decisions fields before inference. Malformed masked JSON and guardrail mock/block responses fail closed without reaching the upstream. Text guardrails cannot inspect audio: a guarded audio request is rejected explicitly rather than silently bypassing the policy. Unguarded audio requests remain supported.

Native Decisions routing estimates text tokens from grounding and question instructions/criteria for usage- and cost-based selection. This estimate does not measure image or audio tokens; actual upstream usage is used for completed-call accounting. Existing SystemOne pass-through routes are unaffected.

Upstream error statuses are preserved, including authentication errors and rate limits with `Retry-After`. Router retry/fallback policy operates on those errors rather than treating every upstream rejection as a connection failure.

Both `POST /v1/decisions` and `POST /decisions` use virtual-key authentication, model routing, request hooks, usage logging and configured pricing. The SDK exposes `litellm.decisions` and `litellm.adecisions` with the same optional `images` parameter.

## Migrate existing clients

Keep the existing registered `/v1/systemone` pass-through for `jev-latest` and `/v1/systemone-custom` pass-through for `jev-trained`. This backport adds no native SystemOne route and changes no pass-through targets, IDs or permissions.

Migrate each client by changing its URL to `/v1/decisions` and its requested model to `xor`, after granting model access. The adapter sends the deployment's configured model (`jev-trained` in the example) to the upstream. Remove legacy pass-through registrations only after their callers have migrated.

The upstream request limits are 128 questions, 255 choice options, and 10 score levels. An upstream wrapper can impose lower limits. The Decisions response retains the SystemOne `answers` and `usage` format.
