import os
import subprocess
import ast


# 2. Tool Definitions
def read_file(filepath, start_line=1, max_lines=100):
    """Reads a file with strict pagination to prevent context window exhaustion."""
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
            content += f"\n\n... [TRUNCATED: Lines {end_idx + 1} to {total_lines} remain. Use read_file with start_line={end_idx + 1} if needed] ..."

        return content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filepath, content):
    """Creates or completely overwrites a file."""
    try:
        if not content.strip():
            return "Error: Refused to write an empty file. If you meant to stop, just announce completion."
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


def append_file(filepath, content):
    """Appends content to the end of an existing file. Perfect for building large files safely."""
    try:
        if not content.strip():
            return "Error: Refused to append empty whitespace. If the file is complete, announce completion and stop."
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist. Use write_file to initialize it first."
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully appended content to {filepath}"
    except Exception as e:
        return f"Error appending to file: {e}"


def patch_file(filepath, search_text, replace_text):
    """Surgically replaces a specific block of text inside a file."""
    try:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist."

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if search_text not in content:
            return "Error: The exact 'search_text' block was not found in the file. Patch failed."

        updated_content = content.replace(search_text, replace_text, 1)  # Only replace first match for safety

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        return f"Successfully patched {filepath}."
    except Exception as e:
        return f"Error patching file: {e}"


def run_cmd(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output[:1200]
    except Exception as e:
        return f"Error executing command: {e}"


def extract_code_blocks(source_filepath, target_filepath, block_names, wrap_in_class=None):
    """
    Deterministically extracts exact source code blocks from the source file
    and writes them to the target file.
    """
    try:
        with open(source_filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)
        extracted_code = []

        # If the LLM wants these methods grouped in a new class
        if wrap_in_class:
            extracted_code.append(f"class {wrap_in_class}:")
            indent_prefix = "    "
        else:
            indent_prefix = ""

        found_blocks = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in block_names:
                    segment = ast.get_source_segment(source, node)
                    if segment:
                        # Fix indentation if we are wrapping top-level methods into a new class
                        if wrap_in_class and not isinstance(node, ast.ClassDef):
                            segment = "\n".join(indent_prefix + line if line.strip() else line
                                                for line in segment.splitlines())

                        extracted_code.append(segment)
                        found_blocks += 1

        if found_blocks == 0:
            return f"Error: None of the requested blocks ({block_names}) were found in the source file."

        # Write the exact code securely to the sandbox
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        with open(target_filepath, 'a', encoding='utf-8') as f:
            f.write("\n\n".join(extracted_code) + "\n\n")

        return f"Success: Extracted {found_blocks} blocks and appended to {os.path.basename(target_filepath)}"

    except Exception as e:
        return f"Extraction Error: {e}"
