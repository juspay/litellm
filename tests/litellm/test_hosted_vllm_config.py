import os
import copy
import pytest
from litellm.llms.hosted_vllm.chat.transformation import HostedVLLMChatConfig


def test_clean_tool_schemas_default():
    """Test that clean_tool_schemas defaults to True"""
    config = HostedVLLMChatConfig()
    assert config.clean_tool_schemas is True


def test_clean_tool_schemas_env_var_false():
    """Test HOSTED_VLLM_CLEAN_TOOL_SCHEMAS env var set to false"""
    os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"] = "false"
    config = HostedVLLMChatConfig()
    assert config.clean_tool_schemas is False
    del os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"]


def test_clean_tool_schemas_env_var_true():
    """Test HOSTED_VLLM_CLEAN_TOOL_SCHEMAS env var set to true"""
    os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"] = "true"
    config = HostedVLLMChatConfig()
    assert config.clean_tool_schemas is True
    del os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"]


def test_clean_tool_schemas_env_var_variations():
    """Test various env var values that should evaluate to false"""
    for false_value in ["0", "no", "False", "FALSE"]:
        os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"] = false_value
        config = HostedVLLMChatConfig()
        assert config.clean_tool_schemas is False, f"Failed for value: {false_value}"
        del os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"]


def test_clean_tool_schemas_constructor_param():
    """Test clean_tool_schemas can be set via constructor"""
    config = HostedVLLMChatConfig(clean_tool_schemas=False)
    assert config.clean_tool_schemas is False


def test_clean_tool_schemas_constructor_overrides_env():
    """Test that constructor parameter overrides environment variable"""
    os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"] = "true"
    config = HostedVLLMChatConfig(clean_tool_schemas=False)
    assert config.clean_tool_schemas is False
    del os.environ["HOSTED_VLLM_CLEAN_TOOL_SCHEMAS"]


def test_map_openai_params_with_clean_enabled():
    """Test that tools are cleaned when clean_tool_schemas=True"""
    config = HostedVLLMChatConfig(clean_tool_schemas=True)

    tools = [{
        "type": "function",
        "function": {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "additionalProperties": False,
                "strict": True
            }
        }
    }]

    non_default_params = {"tools": tools.copy()}
    result = config.map_openai_params(non_default_params, {}, "test-model", False)

    # Verify additionalProperties and strict were removed
    result_params = result["tools"][0]["function"]["parameters"]
    assert "additionalProperties" not in result_params, "additionalProperties should be removed"
    assert "strict" not in result_params, "strict should be removed"


def test_map_openai_params_with_clean_disabled():
    """Test that tools are NOT cleaned when clean_tool_schemas=False"""
    config = HostedVLLMChatConfig(clean_tool_schemas=False)

    tools = [{
        "type": "function",
        "function": {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "additionalProperties": False,
                "strict": True
            }
        }
    }]

    non_default_params = {"tools": tools.copy()}
    result = config.map_openai_params(non_default_params, {}, "test-model", False)

    # Verify additionalProperties and strict are preserved
    result_params = result["tools"][0]["function"]["parameters"]
    assert "additionalProperties" in result_params, "additionalProperties should be preserved"
    assert result_params["additionalProperties"] is False, "additionalProperties value should match"
    assert "strict" in result_params, "strict should be preserved"
    assert result_params["strict"] is True, "strict value should match"


def test_map_openai_params_with_nested_properties():
    """Test that nested additionalProperties are handled correctly"""
    config_clean = HostedVLLMChatConfig(clean_tool_schemas=True)
    config_no_clean = HostedVLLMChatConfig(clean_tool_schemas=False)

    tools = [{
        "type": "function",
        "function": {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"inner": {"type": "string"}},
                        "additionalProperties": False
                    }
                },
                "additionalProperties": False,
                "strict": True
            }
        }
    }]

    # Test with cleaning enabled - use deep copy to avoid mutation
    non_default_params_clean = {"tools": copy.deepcopy(tools)}
    result_clean = config_clean.map_openai_params(non_default_params_clean, {}, "test-model", False)

    # Test with cleaning disabled - use deep copy to avoid mutation
    non_default_params_no_clean = {"tools": copy.deepcopy(tools)}
    result_no_clean = config_no_clean.map_openai_params(non_default_params_no_clean, {}, "test-model", False)

    # With cleaning: nested properties should also be removed
    nested_params_clean = result_clean["tools"][0]["function"]["parameters"]["properties"]["nested"]
    assert "additionalProperties" not in nested_params_clean, "Nested additionalProperties should be removed"

    # Without cleaning: nested properties should be preserved
    nested_params_no_clean = result_no_clean["tools"][0]["function"]["parameters"]["properties"]["nested"]
    assert "additionalProperties" in nested_params_no_clean, "Nested additionalProperties should be preserved"


def test_map_openai_params_without_tools():
    """Test that map_openai_params works correctly when no tools are provided"""
    config = HostedVLLMChatConfig(clean_tool_schemas=True)

    non_default_params = {"temperature": 0.7}
    result = config.map_openai_params(non_default_params, {}, "test-model", False)

    assert "tools" not in result, "tools should not be in result when not provided"
    assert result["temperature"] == 0.7, "other parameters should be preserved"
