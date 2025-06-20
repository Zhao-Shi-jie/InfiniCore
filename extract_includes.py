#!/usr/bin/env python3
import json
import re
import os
from pathlib import Path

def extract_include_paths():
    """从compile_commands.json中提取所有include路径"""
    
    if not os.path.exists('compile_commands.json'):
        print("错误: compile_commands.json 文件不存在")
        return []
    
    include_paths = set()
    
    with open('compile_commands.json', 'r') as f:
        data = json.load(f)
    
    print(f"处理 {len(data)} 个编译条目...")
    
    for entry in data:
        command = entry.get('command', '')
        directory = entry.get('directory', '')
        
        # 提取 -I 参数
        # 匹配 -I/path 或 -I /path 格式
        include_matches = re.findall(r'-I\s*([^\s]+)', command)
        
        for include_path in include_matches:
            # 处理相对路径
            if not os.path.isabs(include_path):
                if directory:
                    include_path = os.path.join(directory, include_path)
                else:
                    include_path = os.path.abspath(include_path)
            
            # 规范化路径
            include_path = os.path.normpath(include_path)
            
            # 检查路径是否存在
            if os.path.exists(include_path):
                include_paths.add(include_path)
    
    return sorted(list(include_paths))

def find_additional_include_dirs():
    """查找项目中的其他潜在include目录"""
    additional_paths = set()
    
    # 查找所有名为include的目录
    for root, dirs, files in os.walk('.'):
        # 跳过一些不需要的目录
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
            continue
            
        for dir_name in dirs:
            if dir_name == 'include':
                full_path = os.path.abspath(os.path.join(root, dir_name))
                additional_paths.add(full_path)
    
    # 查找包含头文件的目录
    header_dirs = set()
    for root, dirs, files in os.walk('.'):
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache']):
            continue
            
        # 如果目录中有.h或.hpp文件，将其添加到列表
        has_headers = any(f.endswith(('.h', '.hpp', '.hh')) for f in files)
        if has_headers:
            full_path = os.path.abspath(root)
            header_dirs.add(full_path)
    
    return sorted(list(additional_paths)), sorted(list(header_dirs))

def generate_clangd_config(include_paths, workspace_root):
    """生成.clangd配置文件"""
    
    # 过滤和分类路径
    project_paths = []
    system_paths = []
    third_party_paths = []
    
    # 首先添加项目根目录，这样可以解析 "include/xxx" 形式的头文件
    project_paths.append("-I.")
    
    for path in include_paths:
        if path.startswith(workspace_root):
            # 转换为相对路径
            rel_path = os.path.relpath(path, workspace_root)
            # 跳过当前目录，因为已经添加了
            if rel_path == '.':
                continue
            
            if 'third_party' in rel_path or '_deps' in rel_path:
                third_party_paths.append(f"-I./{rel_path}")
                # 如果是include目录，也添加其父目录
                if rel_path.endswith('/include'):
                    parent_path = os.path.dirname(rel_path)
                    if parent_path and parent_path != '.':
                        third_party_paths.append(f"-I./{parent_path}")
            else:
                project_paths.append(f"-I./{rel_path}")
                # 如果是include目录，也添加其父目录
                if rel_path.endswith('/include'):
                    parent_path = os.path.dirname(rel_path)
                    if parent_path and parent_path != '.':
                        project_paths.append(f"-I./{parent_path}")
        else:
            system_paths.append(f"-I{path}")
            # 对系统路径也做同样处理
            if path.endswith('/include'):
                parent_path = os.path.dirname(path)
                if parent_path and parent_path != path:
                    system_paths.append(f"-I{parent_path}")
    
    # 去重并排序
    project_paths = sorted(list(set(project_paths)))
    third_party_paths = sorted(list(set(third_party_paths)))
    system_paths = sorted(list(set(system_paths)))
    
    # 生成配置内容
    config_content = f"""# Auto-generated .clangd configuration
# Generated from compile_commands.json

CompileFlags:
  Add:"""
    
    # 添加项目路径
    if project_paths:
        config_content += "\n    # Project include paths"
        for path in sorted(project_paths):
            config_content += f"\n    - {path}"
    
    # 添加第三方库路径
    if third_party_paths:
        config_content += "\n    # Third-party include paths"
        for path in sorted(third_party_paths):
            config_content += f"\n    - {path}"
    
    # 添加系统路径（限制数量以避免过长）
    if system_paths:
        config_content += "\n    # System include paths"
        for path in sorted(system_paths)[:20]:  # 限制前20个系统路径
            config_content += f"\n    - {path}"
    
    config_content += """
  Remove:
    - -W*
    - -fcoroutines-ts

Diagnostics:
  ClangTidy:
    Add: [performance-*, readability-*]
    Remove: [misc-non-private-member-variables-in-classes]
  Suppress: [pp_file_not_found]

Index:
  Background: Build
  Limit: 100000
"""
    
    return config_content

if __name__ == '__main__':
    workspace_root = os.getcwd()
    print(f"工作目录: {workspace_root}")
    
    # 从compile_commands.json提取include路径
    print("\n1. 从compile_commands.json提取include路径...")
    compile_includes = extract_include_paths()
    print(f"   找到 {len(compile_includes)} 个include路径")
    
    # 查找其他include目录
    print("\n2. 查找项目中的其他include目录...")
    additional_includes, header_dirs = find_additional_include_dirs()
    print(f"   找到 {len(additional_includes)} 个名为'include'的目录")
    print(f"   找到 {len(header_dirs)} 个包含头文件的目录")
    
    # 合并所有路径
    all_includes = list(set(compile_includes + additional_includes))
    
    print(f"\n3. 总共找到 {len(all_includes)} 个唯一的include路径")
    
    # 显示前20个路径作为预览
    print("\n预览前20个include路径:")
    for i, path in enumerate(sorted(all_includes)[:20]):
        print(f"   {i+1:2d}. {path}")
    
    if len(all_includes) > 20:
        print(f"   ... 还有 {len(all_includes) - 20} 个路径")
    
    # 生成.clangd配置
    print("\n4. 生成.clangd配置文件...")
    config_content = generate_clangd_config(all_includes, workspace_root)
    
    # 备份现有配置
    if os.path.exists('.clangd'):
        backup_name = '.clangd.backup'
        os.rename('.clangd', backup_name)
        print(f"   已备份现有配置为: {backup_name}")
    
    # 写入新配置
    with open('.clangd', 'w') as f:
        f.write(config_content)
    
    print("   ✓ .clangd配置文件已生成")
    
    # 统计信息
    project_count = len([p for p in all_includes if p.startswith(workspace_root) and 'third_party' not in p and '_deps' not in p])
    third_party_count = len([p for p in all_includes if p.startswith(workspace_root) and ('third_party' in p or '_deps' in p)])
    system_count = len([p for p in all_includes if not p.startswith(workspace_root)])
    
    print(f"\n统计信息:")
    print(f"   项目路径: {project_count}")
    print(f"   第三方路径: {third_party_count}")
    print(f"   系统路径: {system_count}")
    print(f"   总计: {len(all_includes)}")