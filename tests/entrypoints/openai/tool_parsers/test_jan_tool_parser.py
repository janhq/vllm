# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers.jan_tool_parser import JanToolParser
from vllm.tokenizers import TokenizerLike

from .utils import run_tool_extraction

# Test fixtures


@pytest.fixture
def qwen_tokenizer() -> TokenizerLike:
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture
def jan_parser(qwen_tokenizer: TokenizerLike) -> JanToolParser:
    return JanToolParser(qwen_tokenizer)


@pytest.fixture
def chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        seed=42,
        model="Qwen/Qwen3-32B",
        messages=[],
    )


# Unit tests for non-streaming tool call extraction


def test_jan_parser_non_streaming_no_tool_call(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test that plain text without tool calls is handled correctly."""
    text = """This is not a tool call."""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called
    assert tool_call.content == text


def test_jan_parser_non_streaming_tool_call_between_tags(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test tool call extraction with complete opening and closing tags."""
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston", "unit": "celsius"}}
</tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert len(tool_call.tool_calls) == 1
    assert tool_call.tool_calls[0].function.name == "get_current_weather"

    arguments = json.loads(tool_call.tool_calls[0].function.arguments)
    assert arguments["location"] == "Boston"
    assert arguments["unit"] == "celsius"


def test_jan_parser_non_streaming_tool_call_until_eos(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test tool call extraction without closing tag (until end of string)."""
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston"}}"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert len(tool_call.tool_calls) == 1
    assert tool_call.tool_calls[0].function.name == "get_current_weather"

    arguments = json.loads(tool_call.tool_calls[0].function.arguments)
    assert arguments["location"] == "Boston"


def test_jan_parser_non_streaming_tool_call_with_content(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test tool call extraction when there's content before the tool call."""
    text = """Let me check the weather for you.
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "San Francisco"}}
</tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.content == "Let me check the weather for you.\n"
    assert len(tool_call.tool_calls) == 1
    assert tool_call.tool_calls[0].function.name == "get_current_weather"


def test_jan_parser_non_streaming_tool_call_invalid_json(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test that invalid JSON in tool call is handled gracefully."""
    # Missing closing brace to trigger exception
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston"}"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_jan_parser_non_streaming_multiple_tool_calls(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test extraction of multiple tool calls in sequence."""
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston"}}
</tool_call>
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "New York"}}
</tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert len(tool_call.tool_calls) == 2
    assert tool_call.tool_calls[0].function.name == "get_current_weather"
    assert tool_call.tool_calls[1].function.name == "get_current_weather"

    args_0 = json.loads(tool_call.tool_calls[0].function.arguments)
    args_1 = json.loads(tool_call.tool_calls[1].function.arguments)
    assert args_0["location"] == "Boston"
    assert args_1["location"] == "New York"


def test_jan_parser_non_streaming_boolean_parameter(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test tool call with boolean parameter."""
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"

    arguments = json.loads(tool_call.tool_calls[0].function.arguments)
    assert arguments["trigger"] is True


def test_jan_parser_non_streaming_integer_parameter(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test tool call with integer parameter."""
    text = """<tool_call>
{"name": "get_product_info", "arguments": {"product_id": 12345, "quantity": 10}}
</tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "get_product_info"

    arguments = json.loads(tool_call.tool_calls[0].function.arguments)
    assert arguments["product_id"] == 12345
    assert arguments["quantity"] == 10


# Unit tests for streaming tool call extraction


def test_jan_parser_streaming_just_forward_text(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test that plain text without tool calls is streamed correctly."""
    text = """This is some prior text that has nothing to do with tool calling."""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert content == text
    assert len(tool_calls) == 0


def test_jan_parser_streaming_simple_tool_call(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test streaming extraction of a simple tool call."""
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>"""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_current_weather"

    arguments = json.loads(tool_calls[0].function.arguments)
    assert arguments["location"] == "San Francisco"
    assert arguments["unit"] == "celsius"


def test_jan_parser_streaming_tool_call_with_boolean(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test streaming extraction of tool call with boolean parameter."""
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "final_answer"

    arguments = json.loads(tool_calls[0].function.arguments)
    assert arguments["trigger"] is True


def test_jan_parser_streaming_tool_call_with_content(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test streaming when there's text content before the tool call."""
    text = """Let me search for that information.
<tool_call>
{"name": "search_database", "arguments": {"query": "vllm documentation"}}
</tool_call>"""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "search_database"

    # Content before tool call should be preserved
    assert content is not None
    assert "Let me search" in content


def test_jan_parser_streaming_nested_arguments(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test streaming extraction with nested JSON arguments."""
    text = """<tool_call>
{"name": "complex_query", "arguments": {"filters": {"category": "electronics", "price_range": {"min": 100, "max": 500}}, "sort": "relevance"}}
</tool_call>"""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "complex_query"

    arguments = json.loads(tool_calls[0].function.arguments)
    assert arguments["filters"]["category"] == "electronics"
    assert arguments["filters"]["price_range"]["min"] == 100
    assert arguments["filters"]["price_range"]["max"] == 500
    assert arguments["sort"] == "relevance"


def test_jan_parser_streaming_unicode_content(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test streaming with unicode characters in arguments."""
    text = """<tool_call>
{"name": "translate", "arguments": {"text": "Hello 世界", "target_lang": "español"}}
</tool_call>"""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "translate"

    arguments = json.loads(tool_calls[0].function.arguments)
    assert "世界" in arguments["text"]
    assert arguments["target_lang"] == "español"


def test_jan_parser_adjust_request_with_tools(
    jan_parser: JanToolParser,
) -> None:
    """Test that adjust_request correctly sets skip_special_tokens to False when tools are present."""
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )

    adjusted_request = jan_parser.adjust_request(request)
    assert adjusted_request.skip_special_tokens is False


def test_jan_parser_adjust_request_tool_choice_none(
    jan_parser: JanToolParser,
) -> None:
    """Test that adjust_request doesn't modify skip_special_tokens when tool_choice is 'none'."""
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="none",
    )

    adjusted_request = jan_parser.adjust_request(request)
    # When tool_choice is "none", skip_special_tokens should not be set to False
    assert adjusted_request.skip_special_tokens is not False or adjusted_request.skip_special_tokens is None


def test_jan_parser_adjust_request_no_tools(
    jan_parser: JanToolParser,
) -> None:
    """Test that adjust_request doesn't modify request when no tools are present."""
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
    )

    adjusted_request = jan_parser.adjust_request(request)
    # Without tools, the request should pass through parent's adjust_request
    assert adjusted_request is not None


# Edge cases and error handling


def test_jan_parser_empty_tool_call(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test handling of empty tool call tags."""
    text = """<tool_call></tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    # Should not crash, but also should not consider this a valid tool call
    assert not tool_call.tools_called


def test_jan_parser_malformed_tool_tags(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test handling of malformed tool call tags."""
    text = """<tool_call>
{"name": "test"
<tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    # Should handle gracefully without crashing
    assert not tool_call.tools_called


def test_jan_parser_array_arguments(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test tool call with array arguments."""
    text = """<tool_call>
{"name": "batch_process", "arguments": {"items": [1, 2, 3, 4, 5], "operation": "sum"}}
</tool_call>"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "batch_process"

    arguments = json.loads(tool_call.tool_calls[0].function.arguments)
    assert arguments["items"] == [1, 2, 3, 4, 5]
    assert arguments["operation"] == "sum"

def test_jan_parser_streaming_tool_call_with_nested_tag(
    qwen_tokenizer: TokenizerLike,
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test streaming extraction of tool call with boolean parameter."""
    text = """<tool_call>
{"name": "final_answer", "arguments": {"q": "test <tool_call> call"}}
</tool_call>"""

    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "final_answer"

    arguments = json.loads(tool_calls[0].function.arguments)
    assert arguments["q"] == "test <tool_call> call"


def test_jan_parser_non_streaming_multiple_tool_calls_auto_complete(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test extraction of multiple tool calls in sequence then auto complete the missing close bracket."""
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston"}}
</tool_call>
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "New York"}}
</tool_call>
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Los Angeles"}}
</tool_call>
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Chicago"}}
"""
    tool_call = jan_parser.extract_tool_calls(
        model_output=text,
        request=chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert len(tool_call.tool_calls) == 4
    assert tool_call.tool_calls[0].function.name == "get_current_weather"
    assert tool_call.tool_calls[1].function.name == "get_current_weather"
    assert tool_call.tool_calls[2].function.name == "get_current_weather"
    assert tool_call.tool_calls[3].function.name == "get_current_weather"

    args_0 = json.loads(tool_call.tool_calls[0].function.arguments)
    args_1 = json.loads(tool_call.tool_calls[1].function.arguments)
    args_2 = json.loads(tool_call.tool_calls[2].function.arguments)
    args_3 = json.loads(tool_call.tool_calls[3].function.arguments)
    assert args_0["location"] == "Boston"
    assert args_1["location"] == "New York"
    assert args_2["location"] == "Los Angeles"
    assert args_3["location"] == "Chicago"

def test_jan_parser_streaming_multiple_tool_calls_auto_complete(
    jan_parser: JanToolParser,
    chat_request: ChatCompletionRequest,
) -> None:
    """Test extraction of multiple tool calls in sequence with token-by-token streaming, then auto complete the missing close bracket."""
    text = """<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Boston"}}
</tool_call>
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "New York"}
</tool_call>
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Los Angeles"}
"""
    # Use streaming mode to process token by token
    content, tool_calls = run_tool_extraction(
        jan_parser,
        text,
        chat_request,
        streaming=True,
    )

    assert len(tool_calls) == 3
    assert tool_calls[0].function.name == "get_current_weather"
    assert tool_calls[1].function.name == "get_current_weather"
    assert tool_calls[2].function.name == "get_current_weather"

    args_0 = json.loads(tool_calls[0].function.arguments)
    args_1 = json.loads(tool_calls[1].function.arguments)
    args_2 = json.loads(tool_calls[2].function.arguments)
    assert args_0["location"] == "Boston"
    assert args_1["location"] == "New York"
    assert args_2["location"] == "Los Angeles"