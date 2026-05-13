from collections import deque
import torch

class FixedSizeFIFOQueue:
    def __init__(self, max_size):
        self.queue = deque()  # 初始化队列
        self.max_size = max_size  # 固定队列大小

    def enqueue(self, item):
        """添加元素到队列的末尾。如果队列已满，移除最早添加的元素"""
        if len(self.queue) >= self.max_size:
            self.dequeue()  # 移除最早添加的元素
        self.queue.append(item)  # 添加新元素

    def dequeue(self):
        """移除并返回队列的第一个元素"""
        if not self.is_empty():
            return self.queue.popleft()  # 从左侧移除元素
        else:
            raise IndexError("Dequeue from an empty queue")

    def is_empty(self):
        """检查队列是否为空"""
        return len(self.queue) == 0

    def size(self):
        """返回队列的大小"""
        return len(self.queue)

    def peek(self):
        """查看队列的第一个元素而不移除它"""
        if not self.is_empty():
            return self.queue[0]
        else:
            raise IndexError("Peek from an empty queue")

    def to_tensor(self):
        """将队列中的所有元素拼接为一个张量"""
        if self.is_empty():
            raise ValueError("Cannot convert an empty queue to tensor.")
        return torch.cat(list(self.queue), dim=0)  # 在指定维度上拼接
