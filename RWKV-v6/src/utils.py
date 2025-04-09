import json, time, random, os
import numpy as np
import dataclasses
from torch.nn import functional as F
from typing import List, Dict
from io import BytesIO
import base64
from collections import defaultdict

time_slot = {}
time_ref = time.time_ns()

def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


def largest_3n_plus_2_prime(x):
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    # Integer division to obtain an integer n such that 3n+2 < x
    n = x // 3  
    while True:
        num = 3 * n + 2
        if num < x and is_prime(num):
            return num
        n -= 1


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    id: str
    roles: List[str]
    conversations: List[Dict[str, str]]

    def append_message(self, role, message):
        d = {"from": role, "value": message}
        self.conversations.append(d)


def freeze_rwkv_block(model, num_layers_to_freeze):
    # freeze the first num_layers_to_freeze layers
    for i, block in enumerate(model.blocks):
        if i < num_layers_to_freeze:
            for p in block.parameters():
                p.requires_grad_(False)
        else:
            for p in block.parameters():
                p.requires_grad_(True)


def compress_parameter_names(parameter_names):
    compressed = defaultdict(set)
    for weight in parameter_names:
        parts = weight.split('.')
        # find the block number which is a number
        split_index = None
        for i, part in enumerate(parts):
            if part.isdigit():
                block = part
                split_index = i
                break
        if split_index is not None:
            block = parts[split_index]  # 提取block号
            rest = '.'.join(parts[split_index+1:])  # 剩余部分
            prefix = '.'.join(parts[:split_index]) # 
            compressed[(prefix, rest)].add(block)
        else:
            compressed[(weight, '')].add('')

    # 格式化输出，合并具有相同rest部分的block号
    output = []
    for (prefix, rest), blocks in compressed.items():
        if rest and blocks:
            blocks = sorted([int(b) for b in blocks])
            block_range = '{' + ','.join(map(str, blocks)) + '}'
            output.append(f'{prefix}.{block_range}.{rest}')
        else:
            output.append(prefix)
    return output