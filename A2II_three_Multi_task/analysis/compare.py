import pickle

def compare_pkl_files(file1, file2):
    """
    比较两个包含 0/1 列表的 .pkl 文件，返回差异的数量和位置。
    """
    try:
        # 读取第一个 pkl 文件
        with open(file1, "rb") as f:
            list1 = pickle.load(f)
        
        # 读取第二个 pkl 文件
        with open(file2, "rb") as f:
            list2 = pickle.load(f)
        
        # 检查是否为列表
        if not isinstance(list1, list) or not isinstance(list2, list):
            raise ValueError("文件内容不是列表类型（list）")
        
        # 检查列表长度是否一致
        if len(list1) != len(list2):
            raise ValueError("两个列表的长度不一致")
        print(f"列表中共有 {len(list1)} 个值")
        # 检查元素是否为 0/1
        for lst in [list1, list2]:
            for item in lst:
                if item not in {0, 1}:
                    raise ValueError("列表中存在非 0/1 的值")
        
        # 统计不同的元素数量
        differences = sum(1 for a, b in zip(list1, list2) if a != b)
        
        # 扩展：记录差异的具体位置（可选）
        diff_positions = [i for i, (a, b) in enumerate(zip(list1, list2)) if a != b]
        
        return differences, diff_positions
    
    except FileNotFoundError as e:
        print(f"文件不存在: {e}")
    except ValueError as e:
        print(f"数据格式错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

# 示例调用
input_dir='/public/home/ghfu/lzy/code/instructBLIP/A2II_first_work/analysis/twitter_rel_pred'
file1 = f"{input_dir}/our_pred.pkl"
file2 = f"{input_dir}/qformer_pred.pkl"
result = compare_pkl_files(file1, file2)

if result:
    differences, diff_positions = result
    print(f"两个列表中共有 {differences} 个不同值")
    # print("差异位置索引:", diff_positions)  # 可选输出具体位置