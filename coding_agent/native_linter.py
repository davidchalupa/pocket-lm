import ast


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
