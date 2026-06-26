import ast
import os


def analyze_file_metrics(filepath):
    """
    Parses the file using AST to extract the structure and compute
    heuristics to detect a 'God Class' anomaly.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        classes = []
        top_level_functions = 0
        total_methods = 0

        structure_lines = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                classes.append({
                    "name": node.name,
                    "method_count": len(methods)
                })
                total_methods += len(methods)

                method_str = ", ".join(methods) if methods else "No methods"
                structure_lines.append(f"- class {node.name}: {method_str}")

            elif isinstance(node, ast.FunctionDef):
                top_level_functions += 1
                structure_lines.append(f"- top-level function {node.name}()")

        # --- HEURISTIC ENGINE ---
        # Detect if a single class dominates the entire file's structural footprint
        is_god_class = False
        god_class_name = ""

        if classes:
            # Find the largest class
            largest_class = max(classes, key=lambda x: x["method_count"])

            # If a class has more than 12 methods AND holds more than 60% of all functional blocks
            total_blocks = total_methods + top_level_functions
            if largest_class["method_count"] >= 12 and total_blocks > 0:
                share = largest_class["method_count"] / total_blocks
                if share >= 0.60:
                    is_god_class = True
                    god_class_name = largest_class["name"]

        return {
            "structure_map": "\n".join(structure_lines) if structure_lines else "No major structural components.",
            "is_god_class": is_god_class,
            "god_class_name": god_class_name
        }

    except SyntaxError as e:
        return {"error": f"Syntax Error: Cannot parse structure. ({e})"}
    except Exception as e:
        return {"error": f"Error reading file: {e}"}


def build_split_prompt(filepath, target_dir):
    """
    Dynamically generates a token-efficient, framework-agnostic prompt
    optimized for either God Classes or General Monoliths.
    """
    filename = os.path.basename(filepath)
    analysis = analyze_file_metrics(filepath)

    if "error" in analysis:
        return f"System Error: {analysis['error']}"

    structure_map = analysis["structure_map"]

    # Contextual adjustments based on AST diagnostics
    if analysis["is_god_class"]:
        context_framing = (
            f"The file `{filename}` contains an anti-pattern: a single giant 'God Class' "
            f"(`{analysis['god_class_name']}`) which wraps almost all logic."
        )
        architectural_guidance = (
            "- Isolate business logic, data calculations, or heavy processing routines away from core orchestration/UI layers.\n"
            "- Group highly related operations into clean specialized Services, Managers, or specialized sub-components.\n"
            "- Extract secondary UI, operational workflows, or optional attributes into explicit Extension classes or Mixins."
        )
    else:
        context_framing = (
            f"The file `{filename}` is a generic structural monolith with high density and "
            "scattered distinct responsibilities."
        )
        architectural_guidance = (
            "- Group interconnected independent functions and definitions into high-cohesion standalone module domains.\n"
            "- Separate utility primitives, driver logic, or presentation targets into explicit horizontal tiers."
        )

    # Token-optimized structural prompt
    prompt = f"""You are a senior Software Architect. Your task is to design a refactoring split blueprint.

[Context]
File Target: `{filename}`
Layout Footprint:
{context_framing}

[AST Map Output]
{structure_map}

[Architectural Rules]
{architectural_guidance}
- Ensure that core entry execution setups or initialization points remain clearly in the root file.

[Required Output Layout Format]
1. EXPLANATION: Write out your structural design reasoning out loud first. Justify your module division choices.
2. JSON PLAN: At the absolute end of your response, provide exactly one markdown ```json block containing a single valid JSON object mapping recommended filenames to lists of their respective target elements/methods.
"""

    return prompt
