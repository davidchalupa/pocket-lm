import sys
import os
import json
import subprocess
import re
import platform
import ast
from llama_cpp import Llama

# Assumes patch_file has been removed from tool_definitions
from coding_agent.tool_definitions import read_file, write_file, append_file, run_cmd

# 1. Configuration
QWEN_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
CONTEXT_WINDOW = 8192
target_path = QWEN_PATH
loaded_model_name = "Qwen 2.5 Coder 7B (Agent Mode V12 Rewriter+Linter)"


# --- NATIVE LINTER ---
def check_python_syntax(filepath):
    """Natively checks Python files for basic syntax errors."""
    if not filepath.endswith('.py'):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read(), filename=filepath)
        return None
    except SyntaxError as e:
        return f"SyntaxError on line {e.lineno}: {e.msg}\n{e.text}"
    except Exception as e:
        return f"Linter error: {e}"


# --- HYPER-ROBUST PAYLOAD PARSER ---
def parse_robust_tool_call(response_content, tool_json_str):
    payload_match = re.search(r"<payload>(.*?)</payload>", response_content, re.DOTALL)
    raw_payload = payload_match.group(1) if payload_match else None

    json_clean = re.sub(r"<payload>.*?</payload>", "", tool_json_str, flags=re.DOTALL).strip()

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

    # Fallback RegEx Parser
    cleaned = json_clean.strip()
    name_match = re.search(r'"name"\s*:\s*"(write_file|append_file|read_file|run_cmd)"', cleaned)
    if not name_match:
        raise json.JSONDecodeError("Could not isolate tool name signature from model string.", json_clean, 0)

    tool_name = name_match.group(1)
    args = {}

    if tool_name in ["write_file", "append_file"]:
        fp_match = re.search(r'"filepath"\s*:\s*"(.*?)"', cleaned)
        if fp_match: args["filepath"] = fp_match.group(1)

        if raw_payload is not None:
            args["content"] = raw_payload
        else:
            content_match = re.search(r'"content"\s*:\s*"', cleaned)
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
            return f"Error: Resolved path '{abs_target_dir}' is not a valid directory."

        is_windows = platform.system() == "Windows"
        pip_bin = None

        for venv_name in [".venv", "venv", "env"]:
            potential_path = os.path.join(abs_target_dir, venv_name)
            if os.path.isdir(potential_path):
                pip_bin = os.path.join(potential_path, "Scripts", "pip.exe") if is_windows else os.path.join(
                    potential_path, "bin", "pip")
                if os.path.isfile(pip_bin): break

        if not pip_bin:
            return f"Error: No virtual environment found inside '{abs_target_dir}'."

        final_output_path = os.path.join(abs_target_dir, "requirements.txt")
        print(f"   [Backend] Executing: '{pip_bin}' freeze")

        result = subprocess.run(f'"{pip_bin}" freeze', shell=True, capture_output=True, text=True, cwd=abs_target_dir,
                                timeout=15)

        if result.returncode != 0: return f"Error: Pip failed. Stderr: {result.stderr}"

        raw_packages = result.stdout.strip() or "# No dependencies found."
        with open(final_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_packages + "\n")
        return f"SUCCESS: Natively generated target file: '{final_output_path}'"

    except Exception as e:
        return f"Error executing native requirements handler: {e}"


# 4. Initialization Logic
if os.path.exists(target_path):
    print(f"Loading {loaded_model_name} into RAM...")
    try:
        llm = Llama(model_path=target_path, n_ctx=CONTEXT_WINDOW, n_threads=6, n_batch=512, verbose=False)
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        sys.exit(1)
else:
    print(f"❌ Error: Model not found at {target_path}.")
    sys.exit(1)

# 5. System Prompt (Simplified & Aggressive)
SYSTEM_PROMPT = """You are an autonomous coding agent. Solve tasks using tools.

AVAILABLE TOOLS:
1. `read_file`: Reads a file. Args: {"filepath": "<string>", "start_line": <int>, "max_lines": <int>}
2. `write_file`: Overwrites/creates a file ENTIRELY. Args: {"filepath": "<string>"}
3. `append_file`: Appends text to a file. Args: {"filepath": "<string>"}
4. `run_cmd`: Runs a terminal command. Args: {"command": "<string>"}

CRITICAL RULES:
1. NEVER output raw file data inside the JSON block. Always use the `<payload>` tag extension.
2. ONLY output ONE tool call at a time.
3. Never summarize or converse unless the task is completely finished.

FORMAT:
<tool_call>{"name": "write_file", "args": {"filepath": "target.py"}}</tool_call>
<payload>
def example():
    return True
</payload>"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
session_cwd = os.getcwd()

print("\n" + "=" * 60)
print(f"🤖 Local Agent Initialized: [{loaded_model_name}]")
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
        if line.strip() == "/send": break
        if line.strip() == "/quit": sys.exit(0)
        if line.strip() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            session_cwd = os.getcwd()
            print("🧹 Memory cleared!")
            continue
        user_lines.append(line)

    user_input = "\n".join(user_lines).strip()
    if not user_input: continue

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
        if input("Allow this action? (y/n): ").strip().lower() == 'y':
            tool_result = generate_requirements_native(abs_target_dir)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user",
                                                                       "content": f"System Alert: User natively generated requirements for '{abs_target_dir}'. Result: {tool_result}. Acknowledge completion."}]
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
            strategy_steps = "1. Compare Existing README vs Repo Structure.\n2. Use `write_file` to completely rewrite and update the README.\n3. State completion."
        else:
            existing_readme = "No existing README.md found."
            strategy_steps = "1. Evaluate repo structure.\n2. Use `write_file` to create the README from scratch.\n3. State completion."

        hidden_prompt = f"--- CONTEXT ---\nTarget: README.md\n\n--- REPO STRUCTURE ---\n{repo_tree}\n\n--- CURRENT README ---\n{existing_readme}\n\nSTRATEGY:\n{strategy_steps}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": hidden_prompt}]

    else:
        messages.append({"role": "user", "content": user_input})

    # Internal Agent Execution Loop
    while True:
        print(f"\n[Agent]: ", end="", flush=True)
        response_content = ""

        try:
            stream = llm.create_chat_completion(messages=messages, stream=True, temperature=0.1, stop=["</tool_call>"])
            finish_reason = None
            for chunk in stream:
                choice = chunk['choices'][0]
                if choice.get('finish_reason'): finish_reason = choice['finish_reason']
                if 'content' in choice.get('delta', {}):
                    piece = choice['delta']['content']
                    print(piece, end="", flush=True)
                    response_content += piece

            if "<tool_call>" in response_content and "</tool_call>" not in response_content:
                response_content += "</tool_call>"
                print("</tool_call>", end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": response_content})

            tool_json_str = None
            tool_match = re.search(r"<tool_call>(.*?)</tool_call>", response_content, re.DOTALL)
            if tool_match: tool_json_str = tool_match.group(1).strip()

            if tool_json_str:
                try:
                    tool_request = parse_robust_tool_call(response_content, tool_json_str)
                    tool_name = tool_request.get("name")
                    tool_args = tool_request.get("args", {})

                    if "filepath" in tool_args and not os.path.isabs(tool_args["filepath"]):
                        tool_args["filepath"] = os.path.abspath(os.path.join(session_cwd, tool_args["filepath"]))

                    # Interceptor for Empty Payloads
                    if tool_name in ["write_file", "append_file"] and not tool_args.get('content', '').strip():
                        print(f"🛑 [Parser] Blocked empty {tool_name}.")
                        messages.append({"role": "user",
                                         "content": "System Alert: Empty payload. Stop calling tools if task is done."})
                        continue

                    print(f"\n⚠️  AGENT REQUESTS TO EXECUTE: {tool_name}")
                    approval = input("Allow this action? (y/n/edit): ").strip().lower()

                    if approval == 'y':
                        if tool_name == "read_file":
                            tool_result = read_file(tool_args.get("filepath"),
                                                    start_line=tool_args.get("start_line", 1),
                                                    max_lines=tool_args.get("max_lines", 75))
                            tool_reinforcement = ""
                        elif tool_name == "write_file":
                            tool_result = write_file(tool_args.get("filepath"), tool_args.get("content"))

                            # Native Python Linter Check
                            syntax_err = check_python_syntax(tool_args.get("filepath"))
                            if syntax_err:
                                print(f"🐛 [Linter] Syntax Error Caught!")
                                tool_result += f"\n\nCRITICAL LINTING ERROR:\n{syntax_err}\nYou must fix this syntax error using write_file."
                                tool_reinforcement = ""
                            else:
                                tool_reinforcement = "\n\n(System Rule: Write successful. Do NOT output the file's contents. If done, state 'Task Complete' and stop.)"

                        elif tool_name == "append_file":
                            tool_result = append_file(tool_args.get("filepath"), tool_args.get("content"))
                            tool_reinforcement = "\n\n(System Rule: Append successful. If done, state 'Task Complete' and stop.)"
                        elif tool_name == "run_cmd":
                            tool_result = run_cmd(tool_args.get("command"))
                            tool_reinforcement = ""
                        print(f"⚙️  Tool execution finished.")

                    elif approval == 'edit':
                        tool_result = f"User feedback: {input('Provide feedback: ')}"
                        tool_reinforcement = ""
                    else:
                        tool_result = "Action blocked by user."
                        tool_reinforcement = ""

                    messages.append({"role": "user", "content": f"Tool Result:\n{tool_result}{tool_reinforcement}"})
                    continue

                except json.JSONDecodeError as e:
                    print(f"\n❌ [Parser] Syntax error caught.")
                    messages.append(
                        {"role": "user", "content": f"Formatting Failure: {str(e)}\nOutput exactly as requested."})
                    break
            break
        except Exception as e:
            print(f"\n[Error]: {e}")
            break
