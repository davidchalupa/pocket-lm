import ast


class DependencyChecker(ast.NodeVisitor):
    def __init__(self):
        self.defined_names = set()
        self.used_names = set()

    def visit_Import(self, node):
        for alias in node.names:
            # e.g., import os -> 'os'
            name = alias.asname if alias.asname else alias.name
            self.defined_names.add(name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            # e.g., from datetime import datetime -> 'datetime'
            name = alias.asname if alias.asname else alias.name
            self.defined_names.add(name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_arg(self, node):
        # Captures function/method arguments (self, filepath, order_data, etc.)
        self.defined_names.add(node.arg)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        # Captures exception aliases (e.g., the 'e' in 'except Exception as e')
        if node.name:
            self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        # We only care about names being loaded (used)
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            # If a variable is assigned a value locally, it's defined
            self.defined_names.add(node.id)
        self.generic_visit(node)


def check_python_syntax_and_imports(filepath):
    """Checks Python files for syntax errors AND missing/forgotten imports."""
    if not filepath.endswith('.py'):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=filepath)

        # Run our dependency analysis
        checker = DependencyChecker()
        checker.visit(tree)

        # Add built-ins so they aren't marked as missing
        import builtins
        builtin_names = set(dir(builtins))

        # Missing names = Used - Defined - Builtins
        missing_names = checker.used_names - checker.defined_names - builtin_names

        if missing_names:
            return f"Linter Error: The following names/modules are used but never imported or defined: {list(missing_names)}"

        return None
    except SyntaxError as e:
        return f"SyntaxError on line {e.lineno}: {e.msg}\n{e.text}"
    except Exception as e:
        return f"Linter error: {e}"
