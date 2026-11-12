"""
Repro for GLM (glm-private-claw) output-boundary corruption over /v1/messages:

Bug 1: the model batches MULTIPLE tool calls into a SINGLE OpenAI streaming
       chunk (delta.tool_calls = [tc0, tc1, ...]). The adapter's content-block
       classifier only inspects tool_calls[0] while the delta translator loops
       over ALL tool_calls, concatenating every function.arguments into ONE
       tool_use block's partial_json -> `{...}{...}` seam -> client JSON parse
       error ("Extra data").

Bug 2: at the reasoning->answer boundary GLM emits ONE chunk carrying both the
       reasoning tail (reasoning_content) AND the first answer token (content).
       If reasoning_content preempts content, the first token of the text block
       is silently dropped (prose starts mid-word / with a leading space).

These tests are written to FAIL on the broken adapter and PASS once fixed.
"""

import json
from typing import List

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class MockSyncStream:
    def __init__(self, chunks: List[ModelResponseStream]):
        self._chunks = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


def _stop_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        choices=[StreamingChoices(delta=Delta(content=""), index=0, finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _batched_two_tool_chunk() -> ModelResponseStream:
    """One chunk carrying TWO complete tool calls (GLM parallel-call batch)."""
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(
                    role="assistant",
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id="call_a",
                            function=Function(name="TaskCreate", arguments='{"title":"A"}'),
                            type="function",
                            index=0,
                        ),
                        ChatCompletionDeltaToolCall(
                            id="call_b",
                            function=Function(name="TaskCreate", arguments='{"title":"B"}'),
                            type="function",
                            index=1,
                        ),
                    ],
                ),
                index=0,
                finish_reason=None,
            )
        ],
    )


def _collect(wrapper) -> List[dict]:
    return [e for e in wrapper]


def _tool_blocks(events: List[dict]) -> List[dict]:
    return [
        e["content_block"]
        for e in events
        if e.get("type") == "content_block_start"
        and isinstance(e.get("content_block"), dict)
        and e["content_block"].get("type") == "tool_use"
    ]


def _partial_json_by_block(events: List[dict]):
    """Reconstruct each tool_use block's accumulated partial_json, keyed by block index."""
    acc = {}
    for e in events:
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta":
            idx = e["index"]
            acc.setdefault(idx, "")
            acc[idx] += e["delta"].get("partial_json", "")
    return acc


# --------------------------------------------------------------------------
# Bug 1 — batched parallel tool calls in a single chunk
# --------------------------------------------------------------------------
def test_batched_parallel_tool_calls_produce_two_wellformed_blocks():
    chunks = [_batched_two_tool_chunk(), _stop_chunk()]
    wrapper = AnthropicStreamWrapper(completion_stream=MockSyncStream(chunks), model="glm-private-claw")
    events = _collect(wrapper)

    blocks = _tool_blocks(events)
    assert len(blocks) == 2, f"expected 2 tool_use blocks, got {len(blocks)}: {blocks}"

    acc = _partial_json_by_block(events)
    # Each block's accumulated JSON must parse on its own — no `}{` seam.
    for idx, raw in acc.items():
        assert raw, f"block {idx} had empty partial_json"
        json.loads(raw)  # raises json.JSONDecodeError (Extra data) if seam present

    joined = {json.loads(v)["title"] for v in acc.values()}
    assert joined == {"A", "B"}, f"tool args mismatched/merged: {acc}"


# --------------------------------------------------------------------------
# Bug 2 — mixed reasoning tail + first answer token in one chunk
# --------------------------------------------------------------------------
def test_mixed_reasoning_and_content_chunk_keeps_first_token():
    chunks = [
        ModelResponseStream(
            choices=[StreamingChoices(delta=Delta(reasoning_content="Let me think", content="", role="assistant"), index=0, finish_reason=None)],
        ),
        # Boundary chunk: reasoning tail AND first answer token together.
        ModelResponseStream(
            choices=[StreamingChoices(delta=Delta(reasoning_content=" done.", content="Hello"), index=0, finish_reason=None)],
        ),
        ModelResponseStream(
            choices=[StreamingChoices(delta=Delta(content=" world"), index=0, finish_reason=None)],
        ),
        _stop_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=MockSyncStream(chunks), model="glm-private-claw")
    events = _collect(wrapper)

    text = "".join(
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    )
    assert text == "Hello world", f"first answer token dropped; got {text!r}"
