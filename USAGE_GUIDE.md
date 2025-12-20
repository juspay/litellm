# Fix for "No tool calls but found tool output" Error in LiteLLM

## Problem
When calling deepseek models through a vLLM endpoint via LiteLLM, you get:
```
Hosted_vllmException - No tool calls but found tool output
```

Direct vLLM API calls work fine, but LiteLLM calls fail.

## Root Cause
LiteLLM's `hosted_vllm` provider transforms tool schemas by removing `additionalProperties` and `strict` fields. Some vLLM instances have strict validation that requires these fields, causing validation errors.

## Solution Implemented

### Option 1: Environment Variable (Recommended)
```bash
# Disable tool schema cleaning
export HOSTED_VLLM_CLEAN_TOOL_SCHEMAS=false
```

```python
# Or in Python
import os
os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"] = "false"

import litellm
response = litellm.completion(
    model="your-deepseek-model",
    messages=[...],
    api_base="your-vllm-endpoint",
    tools=[...]
)
```

### Option 2: Force OpenAI Provider (Simpler)
```python
import litellm
response = litellm.completion(
    model="your-model",
    messages=[...],
    api_base="your-vllm-endpoint",
    custom_llm_provider="openai",  # Force OpenAI provider
    tools=[...]
)
```

### Option 3: Debug Provider Detection
```python
import litellm
litellm.set_verbose = True
# Run your completion call
# Check logs for: 'Determined provider: hosted_vllm'
```

## Code Changes Made

Modified `litellm/llms/hosted_vllm/chat/transformation.py`:

```python
class HostedVLLMChatConfig(OpenAIGPTConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configuration option to disable tool schema cleaning
        # Some vLLM instances have strict validation that requires
        # 'additionalProperties' and 'strict' fields in tool schemas
        # Can be controlled via environment variable HOSTED_VLLM_CLEAN_TOOL_SCHEMAS
        import os
        env_clean_tool_schemas = os.getenv("HOSTED_VLLM_CLEAN_TOOL_SCHEMAS", "true").lower()
        self.clean_tool_schemas = kwargs.get("clean_tool_schemas",
                                            env_clean_tool_schemas not in ["false", "0", "no"])

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        _tools = non_default_params.pop("tools", None)
        if _tools is not None and self.clean_tool_schemas:  # <-- Only clean if enabled
            # remove 'additionalProperties' from tools
            _tools = _remove_additional_properties(_tools)
            # remove 'strict' from tools
            _tools = _remove_strict_from_schema(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
```

## Alternative Workarounds

### 1. Monkey-patch (Advanced)
```python
import litellm
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig

# Monkey-patch to disable tool cleaning
original_map_openai_params = HostedVLLMChatConfig.map_openai_params

def patched_map_openai_params(self, non_default_params, optional_params, model, drop_params):
    # Skip tool cleaning entirely
    _tools = non_default_params.get("tools")
    if _tools is not None:
        non_default_params["tools"] = _tools
    return super(HostedVLLMChatConfig, self)._map_openai_params(
        non_default_params, optional_params, model, drop_params
    )

HostedVLLMChatConfig.map_openai_params = patched_map_openai_params

# Now use LiteLLM as normal
response = litellm.completion(...)
```

### 2. Check Message Sequence
Ensure your message history doesn't have tool call/tool output mismatches:

```python
# Valid sequence:
messages = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "user", "content": None, "tool_outputs": [...]}  # Must have tool_outputs
]

# Invalid sequence (causes error):
messages = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    # Missing tool_outputs in next user message!
    {"role": "user", "content": "Continue please"}
]
```

## Testing

To verify the fix works:

1. Set the environment variable:
   ```bash
   export HOSTED_VLLM_CLEAN_TOOL_SCHEMAS=false
   ```

2. Run your LiteLLM completion call

3. The error should no longer occur

## Why This Works

1. **Direct vLLM calls work** → Raw request format is valid
2. **LiteLLM transforms tool schemas** → Removes `additionalProperties` and `strict`
3. **Some vLLM instances require these fields** → Validation fails
4. **Disabling cleaning preserves original schemas** → Works like direct calls

The fix allows you to disable LiteLLM's tool schema transformations, making LiteLLM behave like direct vLLM API calls.