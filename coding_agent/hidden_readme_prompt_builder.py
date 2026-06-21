import os

from .tool_definitions import read_file


def build_hidden_readme_prompt(abs_target_dir, repo_tree, existing_readme, strategy_steps):
    """
    Scaffolding function for building a hidden README prompt.
    """
    return (
        f"The user wants to evaluate and maintain a clean, high-quality documentation README file.\n\n"
        f"--- CONTEXT ---\n"
        f"Target Directory: '{abs_target_dir}'\n"
        f"Target File: README.md (Use exactly this relative filename in your tool calls)\n\n"
        f"--- CURRENT REPOSITORY STRUCTURE ---\n{repo_tree}\n--------------------------\n\n"
        f"--- ENTIRE EXISTING README CONTENT ---\n{existing_readme}\n--------------------------\n\n"
        f"STRATEGY:\n{strategy_steps}\n\n"
        f"CRITICAL: Do not call tools with empty arguments or empty payloads."
    )

def build_strategy_steps(readme_path, allow_patch):
    if os.path.exists(readme_path):
        tool_choice = "`patch_file` or `write_file`" if allow_patch else "ONE `write_file`"
        return (
            f"1. ANALYSIS PHASE: Begin your response with a bulleted list comparing the files mentioned in the Existing README against the Current Repository Structure.\n"
            f"2. UPDATE PHASE: If discrepancies exist, execute {tool_choice} tool call to fix the README.\n"
            f"3. COMPLETION (CRITICAL): Once your tool call executes successfully, or if no updates are needed, your task is complete. Output a short text confirmation and DO NOT invoke any further tools."
        )
    else:
        return (
            f"1. Evaluate the repository structure below to infer the overall concept of the project.\n"
            f"2. Use `write_file` along with the `<payload>` block to initialize the README file from scratch.\n"
            f"3. Focus on functional purpose and setup. Exclude trivial boilerplate.\n"
            f"4. COMPLETION (CRITICAL): After the file is written, output a final conversational message announcing completion and DO NOT invoke any further tools."
        )
