import sys
import os
import json
import subprocess
import re
import platform
from llama_cpp import Llama

# 1. Configuration
QWEN_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
CONTEXT_WINDOW = 4096
target_path = QWEN_PATH
loaded_model_name = "Qwen 2.5 Coder 7B (Agent Mode V6 Final)"


# 2. Tool Definitions
def read_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath, content):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, 'w') as f:
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
        return output[:2000]
    except Exception as e:
        return f"Error executing command: {e}"


# 3. Native File Handlers (Bypasses LLM hallucinations)
def generate_requirements_native(target_dir):
    try:
        # Force absolute path resolution
        abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))
        if not os.path.isdir(abs_target_dir):
            return f"Error: Resolved path '{abs_target_dir}' is not a valid system directory."

        # Scan for the local pip executable using the absolute path
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
            cwd=abs_target_dir,  # Enforce shell location context
            timeout=15
        )

        if result.returncode != 0:
            return f"Error: Pip execution failed. Stderr: {result.stderr}"

        raw_packages = result.stdout.strip()
        if not raw_packages:
            return f"Warning: Virtual env found, but it contains 0 installed packages. No file written."

        # Write to the absolute destination path
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
            model_path=target_path,
            n_ctx=CONTEXT_WINDOW,
            n_threads=6,
            n_batch=512,
            verbose=False
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
1. `read_file`: Reads a file from the disk. Args: {"filepath": "<string>"}
2. `write_file`: Writes/overwrites a file. Args: {"filepath": "<string>", "content": "<string>"}
3. `run_cmd`: Runs a terminal command. Args: {"command": "<string>"}

HOW TO USE TOOLS:
Whenever you need to interact with the system, you must output a JSON block wrapped in <tool_call> tags. 
DO NOT write code or markdown outside of the "content" argument when writing files.

Example format:
<tool_call>
{"name": "read_file", "args": {"filepath": "main.py"}}
</tool_call>

WORKFLOW:
1. Output ONE tool call at a time.
2. Wait for the tool result before taking the next step.
3. When the task is entirely complete, state "TASK COMPLETE"."""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("\n" + "=" * 60)
print(f"🤖 Local Agent Initialized: [{loaded_model_name}]")
print("Shortcuts:")
print("  /requirements                -> Generate requirements.txt for current directory")
print("  /requirements <path_to_repo> -> Generate requirements.txt for a specific repository path")
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

    # --- THE CRITICAL FIX: Direct Native Interception ---
    if user_input.startswith("/requirements"):
        parts = user_input.split(" ", 1)
        target_dir = parts[1].strip() if len(parts) > 1 else "."

        abs_target_dir = os.path.abspath(os.path.expanduser(target_dir))
        print(f"\n[DEBUG] Resolving Destination: '{abs_target_dir}'")

        if not os.path.isdir(abs_target_dir):
            print(f"❌ Error: Target directory '{abs_target_dir}' does not exist.")
            continue

        print(f"\n⚠️  MANUAL OVERRIDE: Generate requirements.txt natively?")
        print(f"Target Directory: {abs_target_dir}")
        approval = input("Allow this action? (y/n): ").strip().lower()

        if approval == 'y':
            # Execute the function completely outside of the LLM context
            tool_result = generate_requirements_native(abs_target_dir)
            print(f"⚙️  Tool execution finished.")
            print(f"🤖 Backend Hook Log: {tool_result}")

            # Inform the model that the task is finished so it stays in sync
            messages.append({"role": "user",
                             "content": f"System Alert: The user manually executed the /requirements macro for '{abs_target_dir}'. Result: {tool_result}. Briefly acknowledge that the task is complete."})
        else:
            print("🛑 Action blocked.")
            continue

    else:
        # Standard free-text conversation passes through to the LLM normally
        messages.append({"role": "user", "content": user_input})

    # Internal Agent Execution Loop
    while True:
        print(f"\n[Agent]: ", end="", flush=True)
        response_content = ""

        try:
            stream = llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=0.1,
                stop=["</tool_call>"]
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
                        print("Content Snippet: [Context preview]")
                    else:
                        print(f"Arguments: {tool_args}")

                    approval = input("Allow this action? (y/n/edit): ").strip().lower()

                    tool_result = ""
                    if approval == 'y':
                        if tool_name == "read_file":
                            tool_result = read_file(tool_args.get("filepath"))
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
                    error_msg = "Error: Invalid JSON format. Please ensure strict JSON formatting."
                    print(f"❌ {error_msg}")
                    messages.append({"role": "user", "content": error_msg})
                    continue

            break

        except Exception as e:
            print(f"\n[Error during generation]: {e}")
            break
