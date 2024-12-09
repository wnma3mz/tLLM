import socket
import time
from typing import List, Optional, Tuple

import psutil


def get_ips() -> List[Optional[str]]:
    local_ips = []
    if_addrs = psutil.net_if_addrs()
    for interface_name in if_addrs:
        for address in if_addrs[interface_name]:
            if address.family == socket.AF_INET:  # AF_INET表示IPv4地址
                ip = address.address
                if ip != "127.0.0.1":
                    local_ips.append(ip)
            # TODO
            # elif address.family == socket.AF_INET6:  # AF_INET6表示IPv6地址
            #     ip = address.address
            #     if ip!= "::1":
            #         local_ips.append(ip)
    return local_ips


def tcp_ping(ip: str, port: int, timeout=5) -> Optional[float]:
    try:
        start_time = time.time()
        if is_ipv6(ip):  # 判断是否为 IPv6 地址
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.settimeout(timeout)
        sock.connect((ip, port))
        end_time = time.time()
        delay = (end_time - start_time) * 1000
        sock.close()
        return delay
    except (socket.timeout, socket.error):
        return None


def tcp_ping_test(ip_list: List[str], port: int, timeout=5, count=3) -> Tuple[Optional[str], Optional[float]]:
    min_delay = None
    min_ip = None

    for ip in ip_list:
        delays = []

        for _ in range(count):
            delay = tcp_ping(ip, port, timeout)
            if delay is not None:
                delays.append(delay)

        if delays and all(delay is not None for delay in delays):  # 确保三次都有延迟值
            min_delay_ip = min(delays)
            if min_delay is None or min_delay_ip < min_delay:
                min_delay = min_delay_ip
                min_ip = ip

    return min_ip, min_delay


def is_ipv6(ip):
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except socket.error:
        return False
