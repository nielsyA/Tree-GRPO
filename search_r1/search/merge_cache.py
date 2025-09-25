import json
import argparse
import sys

def merge_json_files(file1_path, file2_path, output_path):
    """
    将两个JSON文件合并成一个。

    合并逻辑:
    1. 读取 file1 和 file2 的内容为字典。
    2. 使用 dict.update() 方法将 file2 的内容合并到 file1 的内容中。
       - 如果有相同的键，file2 的值会覆盖 file1 的值。
    3. 将合并后的字典保存到 output_path。

    Args:
        file1_path (str): 第一个JSON文件的路径。
        file2_path (str): 第二个JSON文件的路径。
        output_path (str): 合并后输出的JSON文件路径。
    """
    print(f"[*] 开始合并...\n    源文件1: {file1_path}\n    源文件2: {file2_path}")

    # 读取第一个文件
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        if not isinstance(data1, dict):
            print(f"[!] 错误: 文件 '{file1_path}' 的内容不是一个有效的JSON对象 (字典)。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"[!] 错误: 文件未找到 '{file1_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[!] 错误: 文件 '{file1_path}' 不是有效的JSON格式。")
        sys.exit(1)

    # 读取第二个文件
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        if not isinstance(data2, dict):
            print(f"[!] 错误: 文件 '{file2_path}' 的内容不是一个有效的JSON对象 (字典)。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"[!] 错误: 文件未找到 '{file2_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[!] 错误: 文件 '{file2_path}' 不是有效的JSON格式。")
        sys.exit(1)

    # 合并字典
    print("[*] 正在合并数据...")
    merged_cache = data1.copy()  # 从第一个字典的副本开始，以免修改原始数据
    merged_cache.update(data2)   # 将第二个字典的数据更新进来
    
    # 保存合并后的文件 (使用您指定的逻辑)
    print(f"[*] 正在将合并结果保存到: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_cache, f, ensure_ascii=False, indent=2)
        print(f"[+] 成功！合并后的文件已保存为 '{output_path}'")
    except IOError as e:
        print(f"[!] 错误: 无法写入文件 '{output_path}'. 错误信息: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="将两个JSON缓存文件合并为一个。")
    parser.add_argument("file1", help="第一个JSON文件的路径 (基础文件)。")
    parser.add_argument("file2", help="第二个JSON文件的路径 (要合并进来的文件，值会覆盖第一个文件中的同名键)。")
    parser.add_argument("output", help="合并后输出的JSON文件路径。")

    # 解析参数
    args = parser.parse_args()

    # 执行合并函数
    merge_json_files(args.file1, args.file2, args.output)