import sys
import os
from llama_cpp import Llama

# 1. Configuration - Define paths for all 3 models
STARCODER_PATH = "models/starcoder2-7b.Q4_K_M.gguf"
QWEN_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
MINISTRAL_PATH = "models/Ministral-8B-Instruct-2410-Q4_K_M.gguf"

CONTEXT_WINDOW = 4096  # How much text it can remember at once

# --- HARDCODED MODEL CHOICE ---
# Change this variable to "ministral", "qwen", or "starcoder"
SELECTED_MODEL = "ministral"
# ------------------------------

# 2. Initialization Logic
llm = None
loaded_model_name = ""
target_path = ""
is_instruct_model = True  # Flag to determine how we talk to it

# Determine which model to load based on the selection
if SELECTED_MODEL == "ministral":
    target_path = MINISTRAL_PATH
    loaded_model_name = "Ministral 8B"
elif SELECTED_MODEL == "qwen":
    target_path = QWEN_PATH
    loaded_model_name = "Qwen 2.5 Coder 7B"
elif SELECTED_MODEL == "starcoder":
    target_path = STARCODER_PATH
    loaded_model_name = "StarCoder2 7B Base"
    is_instruct_model = False  # StarCoder is a base completion model
else:
    print(f"❌ Error: Unknown model selection '{SELECTED_MODEL}'.")
    sys.exit(1)

# Attempt to load the selected model
if os.path.exists(target_path):
    print(f"Loading {loaded_model_name} at {target_path} into RAM...")
    try:
        # StarCoder and Qwen/Ministral are all Transformers, so we can use a standard,
        # CPU-optimized loading configuration for all of them.
        llm = Llama(
            model_path=target_path,
            n_ctx=CONTEXT_WINDOW,
            n_threads=6,  # Optimal for a 16GB CPU machine (leaves 2 cores for OS)
            n_batch=512,  # Standard batching
            verbose=False
        )
    except Exception as e:
        print(f"❌ Failed to load {loaded_model_name}: {e}")
        sys.exit(1)
else:
    print(f"❌ Error: Model file not found at {target_path}.")
    print("Please check your 'models/' directory or run the download script.")
    sys.exit(1)

# 3. Setup Chat History with a System Prompt (Only used by Instruct models)
messages = [
    {
        "role": "system",
        "content": (
            "You are a concise coding tool. "
            "Rules: 1. Provide ONLY the requested code block if the solution is obvious. "
            "2. No 'Sure, I can help with that' or 'Here is the code' filler. "
            "3. If an explanation is needed, use maximum 2 bullet points. "
            "4. Never repeat the user's prompt."
            # "5. Only provide solution to the problem requested."
        )
    }
]

print("\n" + "=" * 50)
print(f"🤖 Local CLI Initialized: Running [{loaded_model_name}]")
print("Instructions:")
print("- Type or paste your prompt and code.")
print("- Type '/send' on a new, empty line to submit.")
print("- Type '/clear' to wipe history (Instruct only).")
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
        if line.strip() == "/clear":
            messages = [messages[0]]
            print("🧹 Memory cleared!")
            continue

        user_lines.append(line)

    user_input = "\n".join(user_lines).strip()

    if not user_input:
        continue

    print(f"\n[Assistant - {loaded_model_name}]: \n", end="", flush=True)

    # 5. Generate Response (Streaming)
    response_content = ""

    try:
        # --- PATH A: INSTRUCT MODELS (Qwen / Ministral) ---
        if is_instruct_model:
            messages.append({"role": "user", "content": user_input})

            stream = llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=0.2  # Low temp for exact code
            )

            for chunk in stream:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    piece = delta['content']
                    print(piece, end="", flush=True)
                    response_content += piece

            # Save to history
            messages.append({"role": "assistant", "content": response_content})

        # --- PATH B: BASE MODELS (StarCoder2) ---
        else:
            # Reformat the prompt so the AI thinks it is reading a file
            # and needs to write the code immediately below it.
            starcoder_prompt = f"/*\nTask: Refactor or fix the following C++ code based on instructions.\nInstructions: {user_input}\n*/\n\n"

            stream = llm.create_completion(
                prompt=starcoder_prompt,
                stream=True,
                max_tokens=2048,
                temperature=0.2,
                stop=["/*", "Task:", "<file_sep>"]  # Prevent it from writing new fake prompts
            )

            for chunk in stream:
                piece = chunk['choices'][0]['text']
                print(piece, end="", flush=True)

            # Base models don't use chat history well, so we don't save it to `messages`

    except Exception as e:
        print(f"\n[Error during generation]: {e}")

    print()  # Print final newline
