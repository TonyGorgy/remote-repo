import argparse
import subprocess
import sys
import re

def install_module(module_name, python_executable):
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", module_name])
        return True
    except subprocess.CalledProcessError:
        return False

def extract_imports(file_name):
    with open(file_name, 'r') as file:
        file_content = file.read()
        imports = re.findall(r'^(?:from\s+(\S+)|import\s+(\S+))(?:\s+|$)', file_content, re.MULTILINE)
        top_level_modules = {imp[0].split('.')[0] if imp[0] else imp[1].split('.')[0] for imp in imports}
        return list(top_level_modules)

def check_module(module, python_executable):
    try:
        subprocess.check_call([python_executable, "-c", f"import {module}"])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_modules(modules, python_executable):
    for module in modules:
        if check_module(module, python_executable):
            print(f"模块 '{module}' 已存在.")
        else:
            print(f"尝试安装模块: {module}")
            if not install_module(module, python_executable):
                correct_name = input(f"安装 '{module}' 失败。请输入正确的包名，或按 Enter 跳过: ").strip()
                if correct_name:
                    install_module(correct_name, python_executable)

def main():
    parser = argparse.ArgumentParser(description="自动检测和安装 Python 脚本依赖.")
    parser.add_argument("script", help="要检查依赖的 Python 脚本文件名")
    parser.add_argument("-p", "--python-path", help="Python 解释器的路径（可选）", default=sys.executable)
    args = parser.parse_args()

    modules_to_check = extract_imports(args.script)
    check_and_install_modules(modules_to_check, args.python_path)

if __name__ == "__main__":
    main()
    