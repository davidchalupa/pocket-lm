import sys
import os
import json
import subprocess
import re
import platform
from llama_cpp import Llama

# 1. Configuration
QWEN_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
# Increased to 8192. Qwen 2.5 Coder handles this easily and it prevents truncation issues.
CONTEXT_WINDOW = 8192
target_path = QWEN_PATH
loaded_model_name = "Qwen 2.5 Coder 7B (Agent Mode V9 Context-Optimized)"


# 2. Tool Definitions
def read_file(filepath, start_line=1, max_lines=200):
    """Reads a file with smart pagination to prevent context window exhaustion."""
    try:
        if not os.path.isfile(filepath):
            return f"Error: '{filepath}' is not a file."

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = max(0, int(start_line) - 1)
        end_idx = start_idx + max(1, int(max_lines))

        selected_lines = lines[start_idx:end_idx]
        content = "".join(selected_lines)

        if total_lines > end_idx:
            content += f"\n\n... [TRUNCATED: Lines {end_idx + 1} to {total_lines} remain. Use read_file with start_line={end_idx + 1} if you need to read further] ..."

        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath, content):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


def run_cmd(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output[:1500]  # Slightly tightened to conserve space
    except Exception as e:
        return f"Error executing command: {e}"


# --- NATIVE MAPPER: /readme ---
def get_repo_structure(startpath, max_depth=3):
    """Natively builds a tree of the repo, skipping heavy/irrelevant folders."""
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

    return tree_str[:1200]  # Tightened structure ceiling


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
            return f"Error: No virtual environment (.venv/venv/env) found inside '{abs_target_dir}'."

        final_output_path = os.path.join(abs_target_dir, "requirements.txt")
        print(f"   [Backend] Executing: '{pip_bin}' freeze")

        result = subprocess.run(
            f'"{pip_bin}" freeze',
            shell=True,
            capture_output=True,
            text=True,
            cwd=abs_target_dir,
            timeout=15
        )

        if result.returncode != 0:
            return f"Error: Pip execution failed. Stderr: {result.stderr}"

        raw_packages = result.stdout.strip()

        if not raw_packages:
            print("\n   [Notice] The virtual environment has 0 packages installed.")
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

# 5. System Prompt
SYSTEM_PROMPT = """You are an autonomous coding agent operating on the user's local machine. 
You solve tasks by thinking, planning, and using tools.

AVAILABLE TOOLS:
1. `read_file`: Reads specified lines from disk. Args: {"filepath": "<string>", "start_line": <int>, "max_lines": <int>}
2. `write_file`: Writes/overwrites a file. Args: {"filepath": "<string>", "content": "<string>"}
3. `run_cmd`: Runs a terminal command. Args: {"command": "<string>"}

HOW TO USE TOOLS:
Whenever you need to interact with the system, you must output a JSON block wrapped in <tool_call> tags.
Do not waste context space. Be concise.

Example format:
<tool_call>
{"name": "read_file", "args": {"filepath": "main.py", "start_line": 1, "max_lines": 100}}
</tool_call>

WORKFLOW:
1. Output ONE tool call at a time.
2. Wait for the tool result before taking the next step.
3. Keep file reading targeted—do not read huge files entirely if you only need context."""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("\n" + "=" * 60)
print(f"🤖 Local Agent Initialized: [{loaded_model_name}]")
print("Shortcuts:")
print("  /requirements [path] -> Safely generates requirements.txt natively")
print("  /readme [path]       -> Explores repo and creates/updates README.md")
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

        if not os.path.isdir(abs_target_dir):
            print(f"❌ Error: Target directory '{abs_target_dir}' does not exist.")
            continue

        print(f"\n⚠️  MANUAL OVERRIDE: Generate requirements.txt natively?")
        approval = input("Allow this action? (y/n): ").strip().lower()

        if approval == 'y':
            tool_result = generate_requirements_native(abs_target_dir)
            # ISOLATE CONTEXT: Refresh history specifically for this systemic workflow
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"System Alert: The user manually executed /requirements for '{abs_target_dir}'. Result: {tool_result}. Briefly acknowledge task completion."}
            ]
        else:
            print("🛑 Action blocked.")
            continue

    # --- MACRO: /readme ---
    elif user_input.startswith("/readme"):
        parts = user_input.split(" ", 1)
        target_dir = parts[1].strip() if len(parts) > 1 else "."
        abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))

        if not os.path.isdir(abs_target_dir):
            print(f"❌ Error: Target directory '{abs_target_dir}' does not exist.")
            continue

        print(f"\n🔍 Pre-computing repository structure for {abs_target_dir}...")
        repo_tree = get_repo_structure(abs_target_dir)

        readme_path = os.path.join(abs_target_dir, "README.md")
        if os.path.exists(readme_path):
            existing_readme = read_file(readme_path, max_lines=150)  # Constrained reading
            print("   [Notice] Existing README.md found. The Agent will attempt to update it.")
        else:
            existing_readme = "No existing README.md found. You must create one from scratch."
            print("   [Notice] No README.md found. The Agent will draft a new one.")

        hidden_prompt = (
            f"The user wants to write or update a `README.md` for the repository located at '{abs_target_dir}'.\n\n"
            f"--- REPOSITORY STRUCTURE ---\n{repo_tree}\n--------------------------\n\n"
            f"--- EXISTING README.MD (TEMPORARY SNIPPET) ---\n{existing_readme}\n--------------------------\n\n"
            f"YOUR TASK:\n"
            f"1. Review the structure. Use the updated `read_file` tool to strategically read lines from 1 to 2 core source files if needed.\n"
            f"2. Once ready, call `write_file` to save a beautiful README.md to '{readme_path}'.\n"
            f"3. Do not output conversational filler. Execute your plan directly via tool calls."
        )

        # ISOLATE CONTEXT: Prevent previous chat logs from filling the window before a heavy write operation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": hidden_prompt}
        ]

    else:
        messages.append({"role": "user", "content": user_input})

    # Internal Agent Execution Loop
    while True:
        print(f"\n[Agent]: ", end="", flush=True)
        response_content = ""

        try:
            stream = llm.create_chat_completion(
                messages=messages, stream=True, temperature=0.1, stop=["</tool_call>"]
            )

            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    piece = delta['content']
                    print(piece, end="", flush=True)
                    response_content += piece

            if "<tool_call>" in response_content and "</tool_call>" not in response_content:
                response_content += "</tool_call>"
                print("</tool_call>", end="", flush=True)

            print()
            messages.append({"role": "assistant", "content": response_content})

            # Robust JSON Extraction
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
                    tool_request = json.loads(tool_json_str)
                    tool_name = tool_request.get("name")
                    tool_args = tool_request.get("args", {})

                    print(f"\n⚠️  AGENT REQUESTS PERMISSION TO EXECUTE: {tool_name}")
                    if tool_name == "write_file":
                        print(f"File: {tool_args.get('filepath')}")
                        print("Content Snippet: \n" + "-" * 20)
                        print(tool_args.get('content', '')[:400] + "\n...[truncated]\n" + "-" * 20)
                    else:
                        print(f"Arguments: {tool_args}")

                    approval = input("Allow this action? (y/n/edit): ").strip().lower()

                    tool_result = ""
                    if approval == 'y':
                        if tool_name == "read_file":
                            # Handle paginated arguments passed by the model
                            s_line = tool_args.get("start_line", 1)
                            m_lines = tool_args.get("max_lines", 200)
                            tool_result = read_file(tool_args.get("filepath"), start_line=s_line, max_lines=m_lines)
                        elif tool_name == "write_file":
                            tool_result = write_file(tool_args.get("filepath"), tool_args.get("content"))
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

                    messages.append({"role": "user", "content": f"Tool Execution Result:\n{tool_result}"})
                    continue

                except json.JSONDecodeError:
                    error_msg = "Error: Invalid JSON format. Ensure strict JSON formatting."
                    print(f"❌ {error_msg}")
                    messages.append({"role": "user", "content": error_msg})
                    continue

            break

        except Exception as e:
            print(f"\n[Error during generation]: {e}")
            break
