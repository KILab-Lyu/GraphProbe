import argparse
import subprocess
import json
import ast
import os
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from pylint import epylint as lint
import sys


def run_pylint(file_path):
    # 运行 Pylint 并获取 JSON 输出
    pylint_cmd = f"pylint {file_path} --output-format=json"
    process = subprocess.Popen(pylint_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    pylint_output = stdout.decode()

    try:
        pylint_data = json.loads(pylint_output)
        if not pylint_data:
            return 10  # 无错误，最高分
        scores = [item.get('score', 0) for item in pylint_data if 'score' in item]
        if not scores:
            return 0
        average_score = sum(scores) / len(scores)
        return average_score
    except json.JSONDecodeError:
        return 0


def run_radon(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    mi = mi_visit(code, False)
    complexities = cc_visit(code)
    total_complexity = sum([c.complexity for c in complexities])
    avg_complexity = total_complexity / len(complexities) if complexities else 0
    return mi, avg_complexity


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
    score = (docstring_count / total_definitions) * 5 if total_definitions > 0 else 0
    return score


def calculate_comment_score(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)
    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
    score = (comment_lines / total_lines) * 5 if total_lines > 0 else 0
    return score


def analyze_graph_model(file_path):
    """
    分析图模型特有的实现
    """
    with open(file_path, 'r') as f:
        code = f.read()
    tree = ast.parse(code)
    graph_layers = ['ChebConv', 'GCNConv', 'GraphSAGEConv', 'GATConv', 'DiffPool', 'TopKPool']
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
    # 图卷积层评分
    total_graph_layers = len(graph_layers)
    used_count = len(used_layers)
    conv_score = (used_count / total_graph_layers) * 5  # 0-5分
    # 图数据处理评分（简化为是否有数据处理模块）
    data_processing = any(
        isinstance(node, ast.ImportFrom) and 'torch_geometric' in node.module for node in ast.walk(tree))
    data_score = 3 if data_processing else 0  # 0-3分
    # 接口一致性评分（简化为是否有一致的接口调用）
    # 这里假设接口一致性为使用至少两种不同的图卷积层
    interface_consistent = used_count >= 2
    interface_score = 2 if interface_consistent else 0  # 0-2分
    return conv_score + data_score + interface_score  # 总分 0-10分


def calculate_bonus_score():
    """
    计算加分项得分
    """
    bonus_score = 0
    # 自动化测试与持续集成（4分）
    has_tests = os.path.isdir('tests')
    test_score = 4 if has_tests else 0
    bonus_score += test_score
    # 容器化与部署支持（3分）
    has_docker = os.path.isfile('Dockerfile') or any(fname.startswith('Dockerfile') for fname in os.listdir('.'))
    docker_score = 3 if has_docker else 0
    bonus_score += docker_score
    # 高效的内存管理与计算优化（2分）
    # 简化为检查是否使用了GPU加速
    with open('target_model.py', 'r') as f:
        code = f.read()
    gpu_optimization = 'cuda' in code.lower()
    memory_optimization = any(opt in code.lower() for opt in ['pin_memory', 'num_workers'])
    optimization_score = 2 if gpu_optimization or memory_optimization else 0
    bonus_score += optimization_score
    # 丰富的文档与示例（1分）
    has_examples = os.path.isdir('examples') or os.path.isfile('README.md')
    example_score = 1 if has_examples else 0
    bonus_score += example_score
    return bonus_score  # 总分 10分


def calculate_final_score(modular_score, param_score, data_score, graph_specific, bonus):
    """
    计算最终的可扩展性得分
    权重分配：
    - 模块化设计：30%
    - 参数化与配置化：20%
    - 可扩展的数据处理：20%
    - 图模型特有：20%
    - 加分项：10%
    """
    final_score = (
            modular_score * 0.3 +
            param_score * 0.2 +
            data_score * 0.2 +
            graph_specific * 0.2 +
            bonus * 0.1
    )
    return final_score


def main(file_path):
    print(f"分析文件: {file_path}\n")

    # 1. 模块化设计评分
    mi, avg_cc = run_radon(file_path)
    # 维护指数（MI）转换为 0-15 分
    if mi >= 85:
        modular_score = 15
    elif mi >= 65:
        modular_score = 10
    else:
        modular_score = 5
    print(f"模块化设计评分（Maintainability Index）：{modular_score}/15")

    # 2. 参数化与配置化评分
    # 简化为检查是否有配置文件或参数化设计
    with open(file_path, 'r') as f:
        code = f.read()
    has_config = any(cfg in code.lower() for cfg in ['argparse', 'config', 'yaml', 'json'])
    param_score = 10 if has_config else 0  # 0-10分
    print(f"参数化与配置化评分：{param_score}/20")

    # 3. 可扩展的数据处理评分
    # 检查是否有数据预处理模块和分布式支持
    data_preprocessing = any(
        isinstance(node, ast.ImportFrom) and 'torch_geometric' in node.module for node in ast.walk(ast.parse(code)))
    distributed_support = 'torch.distributed' in code.lower()
    data_score = 10 if data_preprocessing else 0
    data_score += 10 if distributed_support else 0  # 0-20分
    print(f"可扩展的数据处理评分：{data_score}/20")

    # 4. 图模型特有的可扩展性评分
    graph_specific_score = analyze_graph_model(file_path)
    print(f"图模型特有可扩展性评分：{graph_specific_score}/10")

    # 5. 加分项评分
    bonus_score = calculate_bonus_score(file_path)
    print(f"加分项评分：{bonus_score}/10")

    # 6. 综合评分
    final_score = calculate_final_score(
        modular_score=modular_score,
        param_score=param_score,
        data_score=data_score,
        graph_specific=graph_specific_score,
        bonus=bonus_score
    )
    print(f"\n综合可扩展性得分：{final_score:.1f}/100")

    # 7. 阈值判断
    if final_score < 60:
        quality = "较差（Poor）"
    elif final_score <= 80:
        quality = "标准（Standard）"
    else:
        quality = "良好（Good）"
    print(f"评估结果：{quality}")
    print("\n阈值设置：")
    print(" - 较差（Poor）：< 60分")
    print(" - 标准（Standard）：60-80分")
    print(" - 良好（Good）：> 80分")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='图模型代码可维护性评分系统')
    parser.add_argument(
        'file',
        nargs="?",  # Makes the argument optional
        default="/home/myzhao/project/CodeRec/GNNProbe/uploads/GETest_model.py",
        type=str,
        help='待分析的代码文件路径'
    )
    args = parser.parse_args()
    main(args.file)
