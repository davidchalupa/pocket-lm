import sys
import os
from llama_cpp import Llama

# 1. Configuration - Define both paths
MINISTRAL_PATH = "models/Ministral-8B-Instruct-2410-Q4_K_M.gguf"
CONTEXT_WINDOW = 4096  # How much text (prompt + code + history) it can remember at once

# 2. Initialization with Fallback Logic
llm = None
loaded_model_name = ""

use_qwen = False

# Attempt Primary: Ministral 8B
if not use_qwen:
    print(f"Loading model at {MINISTRAL_PATH} into RAM...")
    try:
        llm = Llama(
            model_path=MINISTRAL_PATH,
            n_ctx=CONTEXT_WINDOW,
            verbose=False
        )
        loaded_model_name = "Ministral 8B"
    except Exception as e:
        print(f"⚠️ Failed to load Ministral: {e}")
        print("Falling back to secondary model...")
else:
    QWEN_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    if llm is None:
        if os.path.exists(QWEN_PATH):
            print(f"Loading model at {QWEN_PATH} into RAM...")
            try:
                llm = Llama(
                    model_path=QWEN_PATH,
                    n_ctx=CONTEXT_WINDOW,
                    verbose=False
                )
                loaded_model_name = "Qwen 2.5 Coder 7B"
            except Exception as e:
                print(f"❌ Failed to load Qwen: {e}")
                sys.exit(1)
        else:
            print("❌ Error: Neither Ministral nor Qwen model files could be loaded.")
            print("Please check your 'models/' directory.")
            sys.exit(1)

# 3. Setup Chat History with a System Prompt
messages = [
    {
        "role": "system",
        "content": (
            "You are a concise coding tool. "
            "Rules: 1. Provide ONLY the requested code block if the solution is obvious. "
            "2. No 'Sure, I can help with that' or 'Here is the code' filler. "
            "3. If an explanation is needed, use maximum 2 bullet points. "
            "4. Never repeat the user's prompt."
        )
    }
]

print("\n" + "=" * 50)
print(f"🤖 Local CLI Initialized: Running [{loaded_model_name}]")
print("Instructions:")
print("- Type or paste your prompt and code.")
print("- Type '/send' on a new, empty line to submit.")
print("- Type '/quit' to exit.")
print("=" * 50)

# 4. Main Chat Loop
while True:
    print("\n[You]:")

    # Multiline input capture
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

        user_lines.append(line)

    user_input = "\n".join(user_lines).strip()

    if not user_input:
        continue

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    print(f"\n[Assistant - {loaded_model_name}]: ", end="", flush=True)

    # 5. Generate Response (Streaming)
    # We stream the output so you don't have to wait for the whole answer to generate
    # before seeing text on the screen. Critical for CPU inference!
    response_content = ""

    try:
        stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.3  # Lower temperature = more precise code
        )

        for chunk in stream:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                piece = delta['content']
                print(piece, end="", flush=True)
                response_content += piece

    except Exception as e:
        print(f"\n[Error during generation]: {e}")

    print()  # Print final newline

    # Add assistant response to history to maintain context
    messages.append({"role": "assistant", "content": response_content})
