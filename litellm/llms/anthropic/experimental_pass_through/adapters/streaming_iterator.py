# What is this?
## Translates OpenAI call to Anthropic `/v1/messages` format
import json
import traceback
from collections import deque
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Literal, Optional

from litellm import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import AdapterCompletionStreamWrapper

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponseStream


class AnthropicStreamWrapper(AdapterCompletionStreamWrapper):
    """
    - first chunk return 'message_start'
    - content block must be started and stopped
    - finish_reason must map exactly to anthropic reason, else anthropic client won't be able to parse it.
    """

    from litellm.types.llms.anthropic import (
        ContentBlockContentBlockDict,
        ContentBlockStart,
        ContentBlockStartText,
        TextBlock,
    )

    sent_first_chunk: bool = False
    sent_content_block_start: bool = False
    sent_content_block_finish: bool = False
    current_content_block_type: Literal["text", "tool_use", "thinking"] = "text"
    sent_last_message: bool = False
    holding_chunk: Optional[Any] = None
    holding_stop_reason_chunk: Optional[Any] = None
    queued_usage_chunk: bool = False
    current_content_block_index: int = 0
    chunk_queue: deque = deque()  # Queue for buffering multiple chunks
    current_content_block_start: ContentBlockContentBlockDict = TextBlock(
        type="text",
        text="",
    )
    sent_content_block_delta: bool = False  # Track if we sent at least one delta for current block

    def __init__(self, completion_stream: Any, model: str):
        super().__init__(completion_stream)
        self.model = model

    def _create_initial_usage_delta(self) -> UsageDelta:
        """
        Create the initial UsageDelta for the message_start event.

        Initializes cache token fields (cache_creation_input_tokens, cache_read_input_tokens)
        to 0 to indicate to clients (like Claude Code) that prompt caching is supported.

        The actual cache token values will be provided in the message_delta event at the
        end of the stream, since Bedrock Converse API only returns usage data in the final
        response chunk.

        Returns:
            UsageDelta with all token counts initialized to 0.
        """
        return UsageDelta(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def __next__(self):
        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                item_to_return = self.chunk_queue.popleft()
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                return item_to_return

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                item_to_return = self.chunk_queue.popleft()
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                return item_to_return

            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                self.chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                item_to_return = self.chunk_queue.popleft()
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                return item_to_return

            for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                should_start_new_block = self._should_start_new_content_block(chunk)

                if should_start_new_block:
                    # 1. Close the old block if it's still open
                    if not self.sent_content_block_finish:
                        # If we never sent a delta for the old block, send an empty one first
                        if not self.sent_content_block_delta:
                            from litellm.types.llms.anthropic import ContentTextBlockDelta
                            self.chunk_queue.append({
                                "type": "content_block_delta",
                                "index": self.current_content_block_index,
                                "delta": ContentTextBlockDelta(type="text_delta", text="")
                            })
                        self.chunk_queue.append({
                            "type": "content_block_stop",
                            "index": self.current_content_block_index
                        })

                    # 2. Prepare for new block - Clear decks and increment index
                    self._increment_content_block_index()
                    self.sent_content_block_finish = False
                    self.sent_content_block_delta = False  # Reset delta tracker for new block
                    self.holding_chunk = None  # Prevent text-finish leaks into tool-start

                    # 3. Queue the NEW start block
                    self.chunk_queue.append({
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": self.current_content_block_start
                    })

                # 4. TRANSLATE HERE (so it uses the updated index and tool state)
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                if should_start_new_block:
                    # 5. Queue the data chunk immediately after the start block
                    self.chunk_queue.append(processed_chunk)
                    item_to_return = self.chunk_queue.popleft()
                    with open("litellm_crash_log.txt", "a") as log_file:
                        log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                    return item_to_return

                if (
                    processed_chunk["type"] == "message_delta"
                    and self.sent_content_block_finish is False
                ):
                    # Queue both the content_block_stop and the holding chunk
                    self.chunk_queue.append(
                        {
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        }
                    )
                    self.sent_content_block_finish = True
                    self.holding_chunk = processed_chunk
                    item_to_return = self.chunk_queue.popleft()
                    with open("litellm_crash_log.txt", "a") as log_file:
                        log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                    return item_to_return
                elif self.holding_chunk is not None:
                    # Queue both chunks
                    self.chunk_queue.append(self.holding_chunk)
                    self.chunk_queue.append(processed_chunk)
                    self.holding_chunk = None
                    item_to_return = self.chunk_queue.popleft()
                    with open("litellm_crash_log.txt", "a") as log_file:
                        log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                    return item_to_return
                else:
                    # Queue the current chunk
                    self.chunk_queue.append(processed_chunk)
                    item_to_return = self.chunk_queue.popleft()
                    with open("litellm_crash_log.txt", "a") as log_file:
                        log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                    return item_to_return

            # Handle any remaining held chunks after stream ends
            if self.holding_chunk is not None:
                self.chunk_queue.append(self.holding_chunk)
                self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                item_to_return = self.chunk_queue.popleft()
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                return item_to_return

            raise StopIteration
        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                item_to_return = self.chunk_queue.popleft()
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {item_to_return}\n")
                return item_to_return
            if self.holding_chunk is not None:
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {self.holding_chunk}\n")
                return self.holding_chunk
            if not self.sent_last_message:
                self.sent_last_message = True
                msg_stop = {"type": "message_stop"}
                with open("litellm_crash_log.txt", "a") as log_file:
                    log_file.write(f"[DEBUG ADAPTER SENDS]: {msg_stop}\n")
                return msg_stop
            raise StopIteration
        except Exception as e:
            verbose_logger.error(
                "Anthropic Adapter - {}\n{}".format(e, traceback.format_exc())
            )
            raise StopIteration

    async def __anext__(self):  # noqa: PLR0915
        # --- KILL SWITCH ---
        # Prevent re-entering the stream loop if we already intercepted the finish_reason
        if getattr(self, "stream_finished_flag", False) and not self.chunk_queue:
            raise StopAsyncIteration
        # -------------------

        from .transformation import LiteLLMAnthropicMessagesAdapter

        try:
            # Always return queued chunks first
            if self.chunk_queue:
                item_to_return = self.chunk_queue.popleft()
                print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                return item_to_return

            # Queue initial chunks if not sent yet
            if self.sent_first_chunk is False:
                self.sent_first_chunk = True
                self.chunk_queue.append(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_{}".format(uuid.uuid4()),
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": self._create_initial_usage_delta(),
                        },
                    }
                )
                item_to_return = self.chunk_queue.popleft()
                print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                return item_to_return

            if self.sent_content_block_start is False:
                self.sent_content_block_start = True
                self.chunk_queue.append(
                    {
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                item_to_return = self.chunk_queue.popleft()
                print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                return item_to_return

            async for chunk in self.completion_stream:
                if chunk == "None" or chunk is None:
                    raise Exception

                # GUARD: If we've already sent message_stop, skip all further processing
                if self.sent_last_message:
                    print(f"[DEBUG GUARD] Skipping chunk after message_stop was sent", flush=True)
                    continue

                # Check if we need to start a new content block
                should_start_new_block = self._should_start_new_content_block(chunk)

                # --- ANTHROPIC SCHEMA FIX ---
                # If this is the final chunk (has a finish_reason), force should_start_new_block to False.
                # Opening a trailing empty text block when the stop_reason is 'tool_use' crashes the SDK.
                finish_reason_val = None
                if hasattr(chunk, "choices") and chunk.choices:
                    finish_reason_val = getattr(chunk.choices[0], "finish_reason", None)
                    if finish_reason_val is not None:
                        should_start_new_block = False
                        print(f"[DEBUG FINISH_DETECTED] finish_reason={finish_reason_val}, chunk_type={type(chunk).__name__}", flush=True)
                # ----------------------------

                # --- THE ULTIMATE OVERRIDE ---
                # Intercept the finish reason directly from the raw chunk, close the active block,
                # send the stop reason, and immediately break the loop, bypassing the translator entirely.
                if finish_reason_val is not None:
                    self.stream_finished_flag = True  # Set kill-switch flag
                    finish_reason = finish_reason_val
                    print(f"[DEBUG ULTIMATE OVERRIDE] TRIGGERED! finish_reason={finish_reason}, index={self.current_content_block_index}", flush=True)
                    anthropic_stop_reason = "end_turn"
                    if finish_reason == "tool_calls":
                        anthropic_stop_reason = "tool_use"
                    elif finish_reason == "stop":
                        anthropic_stop_reason = "end_turn"
                    elif finish_reason == "length":
                        anthropic_stop_reason = "max_tokens"

                    # 1. Ensure the final block has a delta
                    if not self.sent_content_block_delta:
                        self.chunk_queue.append({
                            "type": "content_block_delta",
                            "index": self.current_content_block_index,
                            "delta": {"type": "text_delta", "text": ""}
                        })

                    # 2. Close the final block
                    if not self.sent_content_block_finish:
                        self.chunk_queue.append({
                            "type": "content_block_stop",
                            "index": self.current_content_block_index
                        })
                        self.sent_content_block_finish = True

                    # 3. Send the message delta (usage goes here if we have it)
                    usage_dict = {"input_tokens": 0, "output_tokens": 0}
                    if getattr(chunk, "usage", None) is not None:
                         usage_dict["input_tokens"] = chunk.usage.prompt_tokens or 0
                         usage_dict["output_tokens"] = chunk.usage.completion_tokens or 0

                    self.chunk_queue.append({
                        "type": "message_delta",
                        "delta": {"stop_reason": anthropic_stop_reason},
                        "usage": usage_dict
                    })

                    # Break the loop. The StopAsyncIteration handler will pop these and close the stream.
                    break
                # -----------------------------

                if should_start_new_block:
                    # 1. Close the old block if it's still open
                    if not self.sent_content_block_finish:
                        # If we never sent a delta for the old block, send an empty one first
                        if not self.sent_content_block_delta:
                            from litellm.types.llms.anthropic import ContentTextBlockDelta
                            self.chunk_queue.append({
                                "type": "content_block_delta",
                                "index": self.current_content_block_index,
                                "delta": ContentTextBlockDelta(type="text_delta", text="")
                            })
                        self.chunk_queue.append({
                            "type": "content_block_stop",
                            "index": self.current_content_block_index
                        })

                    # 2. Prepare for new block - Clear decks and increment index
                    self._increment_content_block_index()
                    self.sent_content_block_finish = False
                    self.sent_content_block_delta = False  # Reset delta tracker for new block
                    self.holding_chunk = None  # Prevent text-finish leaks into tool-start

                    # 3. Queue the NEW start block
                    start_block = self.current_content_block_start.copy()

                    # --- ANTHROPIC SCHEMA COMPLIANCE ---
                    if start_block.get("type") == "tool_use":
                        # Anthropic SDK forbids 'input' in the start block during streaming
                        if "input" in start_block:
                            del start_block["input"]
                        # Anthropic SDK strictly requires IDs to match ^[a-zA-Z0-9_-]{1,64}$
                        if "id" in start_block:
                            import re
                            start_block["id"] = re.sub(r'[^a-zA-Z0-9_-]', '_', start_block["id"])
                    # -----------------------------------

                    self.chunk_queue.append({
                        "type": "content_block_start",
                        "index": self.current_content_block_index,
                        "content_block": start_block
                    })

                # 4. TRANSLATE HERE (so it uses the updated index and tool state)
                processed_chunk = LiteLLMAnthropicMessagesAdapter().translate_streaming_openai_response_to_anthropic(
                    response=chunk,
                    current_content_block_index=self.current_content_block_index,
                )

                # --- SANITATION FILTER ---
                if processed_chunk.get("type") == "content_block_delta":
                    delta = processed_chunk.get("delta", {})
                    delta_type = delta.get("type")
                    current_block_type = self.current_content_block_start.get("type")

                    # 1. Drop interleaved text_deltas if we are strictly inside a tool_use block
                    if current_block_type == "tool_use" and delta_type == "text_delta":
                        continue

                    # 2. Drop interleaved input_json_deltas if we are strictly inside a text block
                    if current_block_type == "text" and delta_type == "input_json_delta":
                        continue

                    # 3. Drop ghost empty text chunks
                    if delta_type == "text_delta" and not delta.get("text"):
                        # If we have queued state transitions (like Stop/Start), flush them to the CLI
                        if self.chunk_queue:
                            item_to_return = self.chunk_queue.popleft()
                            print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                            return item_to_return
                        # Otherwise safely drop this useless chunk and fetch the next one
                        continue

                    # 4. Force the index to obey our strictly managed state
                    processed_chunk["index"] = self.current_content_block_index
                # -------------------------

                # Check if this is a usage chunk and we have a held stop_reason chunk
                if (
                    self.holding_stop_reason_chunk is not None
                    and getattr(chunk, "usage", None) is not None
                ):
                    # Merge usage into the held stop_reason chunk
                    merged_chunk = self.holding_stop_reason_chunk.copy()
                    if "delta" not in merged_chunk:
                        merged_chunk["delta"] = {}

                    # Add usage to the held chunk
                    usage_dict: UsageDelta = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                    }
                    # Add cache tokens if available (for prompt caching support)
                    if hasattr(chunk.usage, "_cache_creation_input_tokens") and chunk.usage._cache_creation_input_tokens > 0:
                        usage_dict["cache_creation_input_tokens"] = chunk.usage._cache_creation_input_tokens
                    if hasattr(chunk.usage, "_cache_read_input_tokens") and chunk.usage._cache_read_input_tokens > 0:
                        usage_dict["cache_read_input_tokens"] = chunk.usage._cache_read_input_tokens
                    merged_chunk["usage"] = usage_dict

                    # Queue the merged chunk and reset
                    self.chunk_queue.append(merged_chunk)
                    self.queued_usage_chunk = True
                    self.holding_stop_reason_chunk = None
                    item_to_return = self.chunk_queue.popleft()
                    print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                    return item_to_return

                # Check if this processed chunk has a stop_reason - hold it for next chunk
                print(f"[DEBUG FLOW] chunk_type={getattr(chunk, 'choices', [None])[0].__class__.__name__ if hasattr(chunk, 'choices') and chunk.choices else 'no_choices'}, processed_type={processed_chunk.get('type')}, should_start_new_block={should_start_new_block}, sent_content_block_finish={self.sent_content_block_finish}, queued_usage_chunk={self.queued_usage_chunk}", flush=True)

                if not self.queued_usage_chunk:
                    # FIX: Handle message_delta FIRST to ensure open blocks are closed
                    if (
                        processed_chunk.get("type") == "message_delta"
                        and self.sent_content_block_finish is False
                    ):
                        print(f"[DEBUG MSG_DELTA] ENTERED HANDLER - index={self.current_content_block_index}, sent_content_block_delta={self.sent_content_block_delta}", flush=True)
                        # If we never sent a delta for this block, send an empty one first
                        if not self.sent_content_block_delta:
                            self.chunk_queue.append({
                                "type": "content_block_delta",
                                "index": self.current_content_block_index,
                                "delta": {"type": "text_delta", "text": ""}
                            })
                        # Queue the stop block
                        self.chunk_queue.append({
                            "type": "content_block_stop",
                            "index": self.current_content_block_index,
                        })
                        self.sent_content_block_finish = True
                        # FIX: Always queue message_delta immediately, don't hold it
                        self.chunk_queue.append(processed_chunk)
                        item_to_return = self.chunk_queue.popleft()
                        print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                        return item_to_return

                    if should_start_new_block:
                        # 5. Queue the data chunk immediately after the start block
                        self.chunk_queue.append(processed_chunk)
                        self.sent_content_block_delta = True  # Mark that we sent a delta
                        item_to_return = self.chunk_queue.popleft()
                        print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                        return item_to_return

                    if self.holding_chunk is not None:
                        # Queue both chunks
                        self.chunk_queue.append(self.holding_chunk)
                        self.chunk_queue.append(processed_chunk)
                        self.sent_content_block_delta = True  # Mark that we sent a delta
                        self.holding_chunk = None
                        item_to_return = self.chunk_queue.popleft()
                        print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                        return item_to_return
                    else:
                        # Queue the current chunk
                        self.chunk_queue.append(processed_chunk)
                        self.sent_content_block_delta = True  # Mark that we sent a delta
                        item_to_return = self.chunk_queue.popleft()
                        print(f"\n[DEBUG ADAPTER SENDS]: {item_to_return}\n", flush=True)
                        return item_to_return

            # Handle any remaining held chunks after stream ends
            if not self.queued_usage_chunk:
                if self.holding_stop_reason_chunk is not None:
                    self.chunk_queue.append(self.holding_stop_reason_chunk)
                    self.holding_stop_reason_chunk = None

                if self.holding_chunk is not None:
                    self.chunk_queue.append(self.holding_chunk)
                    self.holding_chunk = None

            if not self.sent_last_message:
                self.sent_last_message = True
                self.chunk_queue.append({"type": "message_stop"})

            # Return queued items if any
            if self.chunk_queue:
                return self.chunk_queue.popleft()

            raise StopIteration

        except StopIteration:
            # Handle any remaining queued chunks before stopping
            if self.chunk_queue:
                return self.chunk_queue.popleft()
            # Handle any held stop_reason chunk
            if self.holding_stop_reason_chunk is not None:
                return self.holding_stop_reason_chunk
            if not self.sent_last_message:
                self.sent_last_message = True
                return {"type": "message_stop"}
            raise StopAsyncIteration

    def anthropic_sse_wrapper(self) -> Iterator[bytes]:
        """
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        Similar to the Bedrock bedrock_sse_wrapper implementation.

        This wrapper ensures dict chunks are SSE formatted with both event and data lines.
        """
        for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    async def async_anthropic_sse_wrapper(self) -> AsyncIterator[bytes]:
        """
        Async version of anthropic_sse_wrapper.
        Convert AnthropicStreamWrapper dict chunks to Server-Sent Events format.
        """
        async for chunk in self:
            if isinstance(chunk, dict):
                event_type: str = str(chunk.get("type", "message"))
                payload = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
                yield payload.encode()
            else:
                # For non-dict chunks, forward the original value unchanged
                yield chunk

    def _increment_content_block_index(self):
        self.current_content_block_index += 1

    def _should_start_new_content_block(self, chunk: "ModelResponseStream") -> bool:
        """
        Determine if we should start a new content block based on the processed chunk.
        Override this method with your specific logic for detecting new content blocks.

        Examples of when you might want to start a new content block:
        - Switching from text to tool calls
        - Different content types in the response
        - Specific markers in the content
        """
        from .transformation import LiteLLMAnthropicMessagesAdapter

        # Example logic - customize based on your needs:
        # If chunk indicates a tool call
        if chunk.choices[0].finish_reason is not None:
            return False

        (
            block_type,
            content_block_start,
        ) = LiteLLMAnthropicMessagesAdapter()._translate_streaming_openai_chunk_to_anthropic_content_block(
            choices=chunk.choices  # type: ignore
        )

        if block_type != self.current_content_block_type:
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        # For parallel tool calls, we'll necessarily have a new content block
        # if we get a function name since it signals a new tool call
        if block_type == "tool_use" and content_block_start.get("name"):
            self.current_content_block_type = block_type
            self.current_content_block_start = content_block_start
            return True

        return False
