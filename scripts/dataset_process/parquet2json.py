import pyarrow.parquet as pq
import pandas as pd
import json
import numpy as np

def parquet_to_llava_json(parquet_file_path, json_file_path):
    # 读取 Parquet 文件
    table = pq.read_table(parquet_file_path)
    # 转换为 Pandas DataFrame
    df = table.to_pandas()

    # 将 DataFrame 转换为字典列表
    data_list = df.to_dict(orient='records')

    # 遍历数据列表，将 ndarray 转换为列表
    for item in data_list:
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                item[key] = value.tolist()

    # 假设 Parquet 文件中的数据已经符合要求，直接使用 data_list
    # 如果不符合要求，需要在这里进行相应的调整

    # 保存为 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

# 示例使用
parquet_file_path = 'train-00000-of-00001.parquet'
json_file_path = 'output.json'
parquet_to_llava_json(parquet_file_path, json_file_path) 