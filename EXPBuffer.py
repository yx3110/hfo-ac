import random


class ExpBuffer:
    def __init__(self, buffer_size=500000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.cur_size = 0

    def add(self, experience):
        if self.cur_size >= self.buffer_size:
            self.buffer.pop(0)
            self.cur_size -= 1
        self.buffer.append(experience)
        self.cur_size += 1

    def sample(self, size):
        res = []
        for i in range(size):
            res.append(self.buffer[random.randint(0, self.cur_size-1)])
        return res
