# shared_memory.py
import ctypes
from multiprocessing import shared_memory


class RingBuffer:
    # 缓冲区头部结构
    class Header(ctypes.Structure):
        _fields_ = [
            ("write_idx", ctypes.c_uint64),  # 写指针
            ("read_idx", ctypes.c_uint64),  # 读指针
            ("buf_size", ctypes.c_uint64),  # 缓冲区大小
        ]

    def __init__(self, name, size=1024 * 1024):  # 默认1MB
        self.buf_size = size
        self.name = name

        try:
            # 尝试连接到已存在的共享内存
            self.shm = shared_memory.SharedMemory(name=name)
            self.is_creator = False
        except FileNotFoundError:
            # 创建新的共享内存
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=ctypes.sizeof(self.Header) + size)
            self.is_creator = True

            # 初始化头部
            header = self.Header.from_buffer(self.shm.buf)
            header.write_idx = 0
            header.read_idx = 0
            header.buf_size = size

        self.header = self.Header.from_buffer(self.shm.buf)
        self.buffer = memoryview(self.shm.buf)[ctypes.sizeof(self.Header) :]

    def write(self, data: bytes) -> bool:
        data_size = len(data)
        if data_size > self.buf_size:
            return False

        # 计算可用空间
        write_idx = self.header.write_idx % self.buf_size
        read_idx = self.header.read_idx % self.buf_size

        if write_idx >= read_idx:
            available = self.buf_size - (write_idx - read_idx)
        else:
            available = read_idx - write_idx

        if data_size + 4 > available:  # 4字节用于存储长度
            return False

        # 写入数据长度
        length_bytes = data_size.to_bytes(4, "little")
        for i, b in enumerate(length_bytes):
            self.buffer[(write_idx + i) % self.buf_size] = b

        # 写入数据
        for i, b in enumerate(data):
            self.buffer[(write_idx + 4 + i) % self.buf_size] = b

        # 更新写指针
        self.header.write_idx = (write_idx + 4 + data_size) % self.buf_size
        return True

    def read(self) -> bytes:
        if self.header.read_idx == self.header.write_idx:
            return None

        read_idx = self.header.read_idx % self.buf_size

        # 读取数据长度
        length_bytes = bytes(self.buffer[read_idx : read_idx + 4])
        data_size = int.from_bytes(length_bytes, "little")

        # 读取数据
        data = bytearray()
        for i in range(data_size):
            data.append(self.buffer[(read_idx + 4 + i) % self.buf_size])

        # 更新读指针
        self.header.read_idx = (read_idx + 4 + data_size) % self.buf_size
        return bytes(data)

    def close(self):
        self.shm.close()
        if self.is_creator:
            self.shm.unlink()
