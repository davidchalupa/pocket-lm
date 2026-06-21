import sys
import os
import json
import subprocess
import re
import platform
from llama_cpp import Llama

from coding_agent.tool_definitions import read_file, write_file, append_file, patch_file, run_cmd
from coding_agent.system_prompt_builder import build_system_prompt
from coding_agent import hidden_readme_prompt_builder

# 1. Configuration
QWEN_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
CONTEXT_WINDOW = 8192  # Ensure maximum headroom for analysis
target_path = QWEN_PATH
loaded_model_name = "Qwen 2.5 Coder 7B (Agent Mode V11 Payload-Safe)"

ALLOW_PATCH = "--allow-patch" in sys.argv


# --- HYPER-ROBUST PAYLOAD PARSER ---
def parse_robust_tool_call(response_content, tool_json_str):
    # FIX: Make the closing </payload> tag optional, or capture up to the end of the string
    payload_match = re.search(r"<payload>(.*?)(?:</payload>|$)", response_content, re.DOTALL)
    raw_payload = payload_match.group(1).strip() if payload_match else None

    # Fallback: If no <payload> tags but there is a markdown code block after the tool call
    if not raw_payload:
        md_block_match = re.search(r"```[a-zA-Z]*\n(.*?)\n```", response_content.split("</tool_call>")[-1], re.DOTALL)
        if md_block_match:
            raw_payload = md_block_match.group(1)

    json_clean = re.sub(r"<payload>.*?(?:</payload>|$)", "", tool_json_str, flags=re.DOTALL).strip()

    try:
        data = json.loads(json_clean, strict=False)
        if "args" not in data:
            data["args"] = {}

        if raw_payload is not None:
            if data.get("name") in ["write_file", "append_file"]:
                data["args"]["content"] = raw_payload
        else:
            if data.get("name") in ["write_file", "append_file"] and "content" not in data["args"]:
                data["args"]["content"] = ""

        return data
    except json.JSONDecodeError:
        pass

    cleaned = json_clean.strip()

    # Dynamically allow patch_file parsing based on flag
    allowed_tools = "write_file|append_file|read_file|run_cmd|patch_file" if ALLOW_PATCH else "write_file|append_file|read_file|run_cmd"
    name_match = re.search(fr'"name"\s*:\s*"({allowed_tools})"', cleaned)

    if not name_match:
        raise json.JSONDecodeError("Could not isolate tool name signature from model string.", json_clean, 0)

    tool_name = name_match.group(1)
    args = {}

    if tool_name in ["write_file", "append_file"]:
        fp_match = re.search(r'"filepath"\s*:\s*"(.*?)"', cleaned)
        if fp_match:
            args["filepath"] = fp_match.group(1)

        if raw_payload is not None:
            args["content"] = raw_payload
        else:
            content_match = re.search(r'"content"\s*:\s*验证"', cleaned) or re.search(r'"content"\s*:\s*"', cleaned)
            if content_match:
                start_idx = content_match.end()
                end_match = re.search(r'"\s*\}\s*\}\s*$', cleaned) or re.search(r'"\s*\}\s*$', cleaned)
                if end_match:
                    args["content"] = cleaned[start_idx:end_match.start()]
                else:
                    raw_tail = cleaned[start_idx:].rstrip(' \n\t}')
                    if raw_tail.endswith('"'): raw_tail = raw_tail[:-1]
                    args["content"] = raw_tail
                args["content"] = args["content"].replace('\\"', '"').replace('\\\\', '\\')
            else:
                args["content"] = ""

        if "filepath" in args:
            return {"name": tool_name, "args": args}

    elif tool_name == "patch_file":
        fp_match = re.search(r'"filepath"\s*:\s*"(.*?)"', cleaned)
        if fp_match:
            args["filepath"] = fp_match.group(1)

        st_match = re.search(r'"search_text"\s*:\s*"', cleaned)
        if st_match:
            start_st = st_match.end()
            end_st_match = re.search(r'",\s*"replace_text"', cleaned)
            if end_st_match:
                args["search_text"] = cleaned[start_st:end_st_match.start()]
            else:
                args["search_text"] = cleaned[start_st:].split('",')[0]
            args["search_text"] = args["search_text"].replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n')

        rt_match = re.search(r'"replace_text"\s*:\s*"', cleaned)
        if rt_match:
            start_rt = rt_match.end()
            end_rt_match = re.search(r'"\s*\}\s*\}\s*$', cleaned) or re.search(r'"\s*\}\s*$', cleaned)
            if end_rt_match:
                args["replace_text"] = cleaned[start_rt:end_rt_match.start()]
            else:
                raw_tail = cleaned[start_rt:].rstrip(' \n\t}')
                if raw_tail.endswith('"'): raw_tail = raw_tail[:-1]
                args["replace_text"] = raw_tail
            args["replace_text"] = args["replace_text"].replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n')

        if "filepath" in args and "search_text" in args and "replace_text" in args:
            return {"name": tool_name, "args": args}

    elif tool_name == "run_cmd":
        cmd_match = re.search(r'"command"\s*:\s*"', cleaned)
        if cmd_match:
            start_idx = cmd_match.end()
            end_match = re.search(r'"\s*\}\s*\}\s*$', cleaned) or re.search(r'"\s*\}\s*$', cleaned)
            if end_match:
                args["command"] = cleaned[start_idx:end_match.start()].replace('\\"', '"').replace('\\\\', '\\')
            else:
                raw_tail = cleaned[start_idx:].rstrip(' \n\t}')
                if raw_tail.endswith('"'): raw_tail = raw_tail[:-1]
                args["command"] = raw_tail.replace('\\"', '"').replace('\\\\', '\\')
            return {"name": tool_name, "args": args}

    elif tool_name == "read_file":
        fp_match = re.search(r'"filepath"\s*:\s*"(.*?)"', cleaned)
        if fp_match: args["filepath"] = fp_match.group(1)
        sl_match = re.search(r'"start_line"\s*:\s*(\d+)', cleaned)
        if sl_match: args["start_line"] = int(sl_match.group(1))
        ml_match = re.search(r'"max_lines"\s*:\s*(\d+)', cleaned)
        if ml_match: args["max_lines"] = int(ml_match.group(1))
        return {"name": tool_name, "args": args}

    raise json.JSONDecodeError("Fallback pattern parser extraction failed.", json_clean, 0)


# --- NATIVE MAPPER: /readme ---
def get_repo_structure(startpath, max_depth=3):
    ignore_dirs = {'.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__', '.idea', '.vscode', 'dist', 'build'}
    tree_str = ""
    start_sep = startpath.count(os.path.sep)

    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        depth = root.count(os.path.sep) - start_sep

        if depth > max_depth:
            del dirs[:]
            continue

        indent = ' ' * 4 * depth
        tree_str += f"{indent}{os.path.basename(root)}/\n"
        subindent = ' ' * 4 * (depth + 1)
        for f in files:
            if not f.startswith('.'):
                tree_str += f"{subindent}{f}\n"

    return tree_str[:1200]


# --- NATIVE HANDLER: /requirements ---
def generate_requirements_native(target_dir):
    try:
        abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))
        if not os.path.isdir(abs_target_dir):
            return f"Error: Resolved path '{abs_target_dir}' is not a valid system directory."

        is_windows = platform.system() == "Windows"
        pip_bin = None

        for venv_name in [".venv", "venv", "env"]:
            potential_path = os.path.join(abs_target_dir, venv_name)
            if os.path.isdir(potential_path):
                if is_windows:
                    test_pip = os.path.join(potential_path, "Scripts", "pip.exe")
                else:
                    test_pip = os.path.join(potential_path, "bin", "pip")

                if os.path.isfile(test_pip):
                    pip_bin = test_pip
                    break

        if not pip_bin:
            return f"Error: No virtual environment found inside '{abs_target_dir}'."

        final_output_path = os.path.join(abs_target_dir, "requirements.txt")
        print(f"   [Backend] Executing: '{pip_bin}' freeze")

        result = subprocess.run(
            f'"{pip_bin}" freeze', shell=True, capture_output=True, text=True, cwd=abs_target_dir, timeout=15
        )

        if result.returncode != 0:
            return f"Error: Pip execution failed. Stderr: {result.stderr}"

        raw_packages = result.stdout.strip()
        if not raw_packages:
            raw_packages = "# No dependencies found. The virtual environment is empty."

        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_packages + "\n")

        return f"SUCCESS: Natively generated target file: '{final_output_path}'"

    except Exception as e:
        return f"Error executing native requirements handler: {e}"


# 4. Initialization Logic
if os.path.exists(target_path):
    print(f"Loading {loaded_model_name} into RAM...")
    try:
        llm = Llama(
            model_path=target_path, n_ctx=CONTEXT_WINDOW, n_threads=6, n_batch=512, verbose=False
        )
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        sys.exit(1)
else:
    print(f"❌ Error: Model file not found at {target_path}.")
    sys.exit(1)

# 5. System Prompt (Dynamically Built)
SYSTEM_PROMPT = build_system_prompt(ALLOW_PATCH)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
session_cwd = os.getcwd()

print("\n" + "=" * 60)
print(f"🤖 Local Agent Initialized: [{loaded_model_name}]")
if ALLOW_PATCH:
    print("🔧 Patching Enabled (--allow-patch active)")
print("Shortcuts:")
print("  /requirements [path] -> Safely generates requirements.txt natively")
print("  /readme [path]       -> Explores repo and builds a modular README.md")
print("Commands: Type /send to submit, /quit to exit, /clear to wipe memory.")
print("=" * 60)

# 6. Main Agent Loop
while True:
    print("\n[You]:")
    user_lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if line.strip() == "/send":
            break
        if line.strip() == "/quit":
            print("Exiting. Goodbye!")
            sys.exit(0)
        if line.strip() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            session_cwd = os.getcwd()
            print("🧹 Memory cleared!")
            continue

        user_lines.append(line)

    user_input = "\n".join(user_lines).strip()
    if not user_input:
        continue

    # --- MACRO: /requirements ---
    if user_input.startswith("/requirements"):
        parts = user_input.split(" ", 1)
        target_dir = parts[1].strip() if len(parts) > 1 else "."
        abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))
        session_cwd = abs_target_dir

        if not os.path.isdir(abs_target_dir):
            print(f"❌ Error: Target directory '{abs_target_dir}' does not exist.")
            continue

        print(f"\n⚠️  MANUAL OVERRIDE: Generate requirements.txt natively?")
        approval = input("Allow this action? (y/n): ").strip().lower()

        if approval == 'y':
            tool_result = generate_requirements_native(abs_target_dir)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"System Alert: User manually ran /requirements for '{abs_target_dir}'. Result: {tool_result}. Briefly acknowledge completion."}
            ]
        else:
            print("🛑 Action blocked.")
            continue

    # --- MACRO: /readme ---
    elif user_input.startswith("/readme"):
        parts = user_input.split(" ", 1)
        target_dir = parts[1].strip() if len(parts) > 1 else "."
        abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))
        session_cwd = abs_target_dir

        if not os.path.isdir(abs_target_dir):
            print(f"❌ Error: Target directory '{abs_target_dir}' does not exist.")
            continue

        print(f"\n🔍 Pre-computing repository structure for {abs_target_dir}...")
        repo_tree = get_repo_structure(abs_target_dir)

        readme_path = os.path.join(abs_target_dir, "README.md")

        if os.path.exists(readme_path):
            existing_readme = read_file(readme_path, start_line=1, max_lines=1000)
            print("   [Notice] Existing README.md found. Forcing structural analysis.")
        else:
            existing_readme = "No existing README.md found. Create from scratch."
            print("   [Notice] No README.md found. Agent will draft a new one focusing on project concept.")

        strategy_steps = hidden_readme_prompt_builder.build_strategy_steps(readme_path, ALLOW_PATCH)

        hidden_readme_prompt = hidden_readme_prompt_builder.build_hidden_readme_prompt(
            abs_target_dir, repo_tree, existing_readme, strategy_steps
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": hidden_readme_prompt}
        ]

    else:
        messages.append({"role": "user", "content": user_input})

    # Internal Agent Execution Loop
    while True:
        print(f"\n[Agent]: ", end="", flush=True)
        response_content = ""

        try:
            stream = llm.create_chat_completion(
                messages=messages, stream=True, temperature=0.1,
                # this seems to cause problems when executing append instructions
                # stop=["</tool_call>"],
            )

            finish_reason = None
            for chunk in stream:
                choice = chunk['choices'][0]
                if choice.get('finish_reason'):
                    finish_reason = choice['finish_reason']

                delta = choice.get('delta')
                if 'content' in delta:
                    piece = delta['content']
                    print(piece, end="", flush=True)
                    response_content += piece

            is_truncated = (finish_reason == "length")

            if "<tool_call>" in response_content and "</tool_call>" not in response_content:
                response_content += "</tool_call>"
                print("</tool_call>", end="", flush=True)

            print()
            messages.append({"role": "assistant", "content": response_content})

            # Extract tool arguments
            tool_json_str = None
            tool_match = re.search(r"<tool_call>(.*?)</tool_call>", response_content, re.DOTALL)
            if tool_match:
                tool_json_str = tool_match.group(1).strip()
            else:
                md_match = re.search(r"```json\s*\n(.*?)\n```", response_content, re.DOTALL)
                if md_match:
                    tool_json_str = md_match.group(1).strip()

            if tool_json_str:
                try:
                    if is_truncated and "write_file" in response_content:
                        raise json.JSONDecodeError("Incomplete payload due to context limit truncation.", tool_json_str,
                                                   0)

                    tool_request = parse_robust_tool_call(response_content, tool_json_str)
                    tool_name = tool_request.get("name")
                    tool_args = tool_request.get("args", {})

                    if "filepath" in tool_args and not os.path.isabs(tool_args["filepath"]):
                        tool_args["filepath"] = os.path.abspath(os.path.join(session_cwd, tool_args["filepath"]))

                    # Hard Interceptor for Empty Payloads on write/append mutations
                    if tool_name in ["write_file", "append_file"]:
                        content = tool_args.get('content', '')
                        if not content.strip():
                            print(f"🛑 [Parser Interceptor] Blocked an empty {tool_name} operation.")
                            messages.append({
                                "role": "user",
                                "content": f"System Alert: You attempted to call {tool_name} with an empty payload. If you have no changes to make, do NOT call a tool. Announce completion instead."
                            })
                            continue

                    print(f"\n⚠️  AGENT REQUESTS PERMISSION TO EXECUTE: {tool_name}")
                    if tool_name in ["write_file", "append_file"]:
                        print(f"Resolved Target File: {tool_args.get('filepath')}")
                        print("Content Snippet: \n" + "-" * 20)
                        print(tool_args.get('content', '')[:300] + "\n...[truncated snippet]\n" + "-" * 20)
                    elif tool_name == "patch_file":
                        print(f"Patching File: {tool_args.get('filepath')}")
                        print(f"Targeting Code block:\n--->\n{tool_args.get('search_text')}\n<---")
                        print(f"Replacing With:\n--->\n{tool_args.get('replace_text')}\n<---")
                    else:
                        print(f"Arguments: {tool_args}")

                    approval = input("Allow this action? (y/n/edit): ").strip().lower()

                    tool_result = ""
                    tool_reinforcement = ""

                    if approval == 'y':
                        if tool_name == "read_file":
                            s_line = tool_args.get("start_line", 1)
                            m_lines = tool_args.get("max_lines", 75)
                            tool_result = read_file(tool_args.get("filepath"), start_line=s_line, max_lines=m_lines)
                        elif tool_name == "write_file":
                            tool_result = write_file(tool_args.get("filepath"), tool_args.get("content"))
                            tool_reinforcement = "\n\n(System Rule: Write successful. Do NOT output the file's contents. If your primary task is complete, state 'Task Complete' in plain text and STOP calling tools. Wait for the user.)"
                        elif tool_name == "append_file":
                            tool_result = append_file(tool_args.get("filepath"), tool_args.get("content"))
                            tool_reinforcement = "\n\n(System Rule: Append successful. If your primary task is complete, state 'Task Complete' in plain text and STOP calling tools. Wait for the user.)"
                        elif tool_name == "patch_file":
                            tool_result = patch_file(tool_args.get("filepath"), tool_args.get("search_text"),
                                                     tool_args.get("replace_text"))
                            tool_reinforcement = "\n\n(System Rule: Patch successful. Do not summarize. If your primary task is complete, state 'Task Complete' in plain text and STOP calling tools. Wait for the user.)"
                        elif tool_name == "run_cmd":
                            tool_result = run_cmd(tool_args.get("command"))
                        else:
                            tool_result = "Error: Unknown tool."
                        print(f"⚙️  Tool execution finished.")

                    elif approval == 'edit':
                        feedback = input("Provide feedback or correction to the agent: ")
                        tool_result = f"User denied the action and provided this feedback: {feedback}"
                    else:
                        tool_result = "User denied permission to execute this tool."
                        print("🛑 Action blocked by user.")

                    messages.append(
                        {"role": "user", "content": f"Tool Execution Result:\n{tool_result}{tool_reinforcement}"})
                    continue

                except json.JSONDecodeError as e:
                    error_msg = (
                        f"Formatting Failure: {str(e)}\n"
                        "Your block string parsing crashed. Remember to output using the minified raw tag:\n"
                        "<tool_call>{\"name\": \"write_file\", \"args\": {\"filepath\": \"target_file.md\"}}</tool_call>\n"
                        "<payload>\nRAW UNESCAPED CONTENT HERE\n</payload>"
                    )
                    print(f"\n❌ [Parser Interceptor] Halted a syntax loop. Returning loop control to user.")
                    messages.append({"role": "user", "content": error_msg})
                    break

            break

        except Exception as e:
            print(f"\n[Error during generation]: {e}")
            break
