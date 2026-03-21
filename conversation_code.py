import sys
from llama_cpp import Llama

# 1. Configuration
MODEL_PATH = "models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
CONTEXT_WINDOW = 4096  # How much text (prompt + code + history) it can remember at once

print(f"Loading model into RAM from {MODEL_PATH}...")
print("This might take a few seconds...\n")

# 2. Initialize the Model
# n_ctx sets the context window. 4096 is a good balance for CPU memory.
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CONTEXT_WINDOW,
    verbose=False  # Set to True if you want to see detailed memory/CPU usage stats
)

# 3. Setup Chat History with a System Prompt
# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful, brilliant software engineering assistant. Provide clear, concise explanations and output well-formatted code."
#     }
# ]
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

print("=" * 50)
print("🤖 Local Qwen 2.5 Coder CLI Initialized")
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

    print("\n[Assistant]: ", end="", flush=True)

    # 5. Generate Response (Streaming)
    # We stream the output so you don't have to wait for the whole answer to generate
    # before seeing text on the screen. Critical for CPU inference!
    response_content = ""

    try:
        stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=0.3  # Lower temperature = more precise/less creative code
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

    # Add assistant response to history to maintain context for the next question
    messages.append({"role": "assistant", "content": response_content})
