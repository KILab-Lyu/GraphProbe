import subprocess
import json
import ast
from radon.complexity import cc_visit
from radon.metrics import mi_visit
import sys
import os
import argparse
# def run_pylint(file_path):
#     pylint_cmd = [sys.executable, '-m', 'pylint', file_path, '--output-format=json']
#
#     try:
#         process = subprocess.Popen(
#             pylint_cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#         stdout, stderr = process.communicate()
#         pylint_output = stdout.decode()
#
#         # Check if pylint produced any output
#         if not pylint_output:
#             print(f"Pylint Error Output:\n{stderr.decode()}")
#             return 0
#
#         pylint_data = json.loads(pylint_output)
#         if not pylint_data:
#             return 10  # No errors, highest score
#
#         severity_weights = {
#             'convention': 1,
#             'refactor': 2,
#             'warning': 3,
#             'error': 4,
#             'fatal': 5
#         }
#
#         total_score = 10  # Start with the highest possible score
#         error_count = 0  # Keep track of the number of errors
#         for msg in pylint_data:
#             msg_type = msg.get('type', 'convention')
#             weight = severity_weights.get(msg_type, 1)
#             total_score -= weight  # Deduct based on severity
#             if weight >= 3:  # Count only significant issues
#                 error_count += 1
#
#         # Add a penalty for excessive errors
#         penalty = min(error_count, 3) * 1  # Cap penalty to avoid too low scores
#         total_score -= penalty
#         # Ensure the score doesn't drop below 0
#         average_score = max(total_score, 0)
#         return average_score
#
#     except subprocess.CalledProcessError as e:
#         print(f"Pylint execution failed: {e}")
#         return 0
#     except json.JSONDecodeError:
#         print("Failed to decode Pylint JSON output.")
#         return 0


def run_radon(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    mi = mi_visit(code, False)
    complexities = cc_visit(code)
    total_complexity = sum([c.complexity for c in complexities])
    avg_complexity = total_complexity / len(complexities) if complexities else 0
    # Introduce a penalty for high complexity
    complexity_score = max(10 - (avg_complexity // 5), 0)  # Score from 0 to 10 based on complexity
    return mi, complexity_score


def calculate_docstring_score(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    tree = ast.parse(code)
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node)
            if doc:
                docstrings.append(doc)
    total_definitions = len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))])
    docstring_count = len(docstrings)
    docstring_score = (docstring_count / total_definitions) * 5 if total_definitions > 0 else 0
    return docstring_score


def calculate_comment_score(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)
    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
    comment_score = (comment_lines / total_lines) * 5 if total_lines > 0 else 0
    return comment_score


def analyze_graph_model(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    tree = ast.parse(code)
    graph_layers = ['ChebConv', 'GCNConv', 'GraphSAGEConv', 'GATConv', 'DiffPool', 'TopKPool', 'SAGEConv', 'GraphConv']
    used_layers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                func_name = func.attr
            elif isinstance(func, ast.Name):
                func_name = func.id
            else:
                continue
            if func_name in graph_layers:
                used_layers.add(func_name)
    conv_score = (len(used_layers) / len(graph_layers)) * 5  # Updated score for graph layers
    data_processing = any(
        isinstance(node, ast.ImportFrom) and 'torch_geometric' in node.module for node in ast.walk(tree))
    data_score = 3 if data_processing else 0
    interface_consistent = all(layer in used_layers for layer in ['ChebConv', 'GCNConv'])
    interface_score = 2 if interface_consistent else 0
    return conv_score + data_score + interface_score


import pylint.lint
def getMaintainScore(file_path, model_file="GE_TestModel.py"):
    print(f"分析文件: {file_path}\n")

    readability_score, saved_file = run_pylint(file_path, model_file)
    mi, complexity_score = run_radon(file_path)
    docstring_score = calculate_docstring_score(file_path)
    comment_score = calculate_comment_score(file_path)
    documentation_score = docstring_score + comment_score
    print(f"文档与注释评分（Docstring 覆盖率 + 注释密度）：{documentation_score:.1f}")
    return {
            "Pylint": f"{readability_score:.1f}",
            "radon":  f"{mi}",
            "Doc":    f"{documentation_score:.1f}",
            "Saved_files": f"{saved_file}"
            }


import os
import io
import pylint.lint
from pylint.reporters.text import TextReporter

def run_pylint(file_path, model_file):
    output = io.StringIO()
    reporter = TextReporter(output)
    linter = pylint.lint.Run([file_path], reporter=reporter, exit=False)
    pylint_output = output.getvalue()
    score = linter.linter.stats.global_note
    save_path = '/data2/myzhao/project/CodeRec/GNNProbe/uploads_saved'
    file_name = os.path.join(save_path, model_file + "CodeQuality.txt")
    os.makedirs(save_path, exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(pylint_output)
    return score, file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图模型代码可维护性评分系统')
    parser.add_argument(
        'file',
        nargs="?",  # Makes the argument optional
        default="/home/myzhao/project/CodeRec/GNNProbe/uploads/GETest_model.py",
        type=str,
        help='待分析的代码文件路径'
    )
    output_pdf = 'pylint_report.pdf'
    args = parser.parse_args()
    # run_pylint(args.file)
    model_file = "GETest_model"
    getMaintainScore("/home/myzhao/project/CodeRec/GNNProbe/uploads/GETest_model.py", model_file)
