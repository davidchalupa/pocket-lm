
def build_system_prompt(allow_patch=False):
    """
    Builds and returns a system prompt for a language model.
    """
    # 5. System Prompt (Dynamically Built)
    tools_section = (
        '4. `patch_file`: {"filepath": "<str>", "search_text": "<str>", "replace_text": "<str>"}\n5. `run_cmd`: {"command": "<str>"}'
        if allow_patch else
        '4. `run_cmd`: {"command": "<str>"}'
    )

    rule_5 = "" if allow_patch else "\n5. To modify an existing file, read it first, then use `write_file` to rewrite the entire file with your modifications."

    return f"""You are a local autonomous coding agent. Use tools modularly to solve tasks.

    AVAILABLE TOOLS:
    1. `read_file`: {{"filepath": "<str>", "start_line": <int>, "max_lines": <int>}}
    2. `write_file`: {{"filepath": "<str>"}} - Overwrites or initializes a file completely with new contents.
    3. `append_file`: {{"filepath": "<str>"}}
    {tools_section}

    CRITICAL RULES:
    1. Output EXACTLY ONE tool call per response wrapped in `<tool_call>` tags, then wait for results.
    2. The JSON tool call MUST be minified on a SINGLE LINE.
    3. NEVER pass raw file data inside JSON. ALWAYS put file content inside a `<payload>` tag immediately following the JSON.
    4. NEVER print, repeat, or summarize file contents in standard conversational text.{rule_5}

    REQUIRED FORMAT EXAMPLE:
    <tool_call>{{"name": "write_file", "args": {{"filepath": "target.py"}}}}
    <payload>
    def sample_function():
        print("Literal, unescaped content goes here!")
    </payload></tool_call>"""
