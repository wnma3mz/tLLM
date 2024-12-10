import ipaddress
import socket
import time
from typing import List, Optional, Tuple

import psutil


def score_ip(ip_info: Tuple[str, dict]) -> int:
    ip, info = ip_info
    score = 0

    # 优先选择IPv4
    # if info['type'] == 'ipv4':
    #     score += 1000

    # 优先选择私有网络地址
    # if info['is_private']:
    #     score += 500

    # 考虑网络速度
    score += info["speed"]

    # 排除特殊地址
    if ip.startswith("169.254."):  # 链路本地地址
        score -= 2000

    return score


def get_ips() -> List[Optional[str]]:
    ip_info = []
    network_interfaces = psutil.net_if_addrs()
    net_stats = psutil.net_if_stats()

    for interface, addrs in network_interfaces.items():
        if interface in net_stats:
            stats = net_stats[interface]
            # 跳过非活跃接口
            if not stats.isup:
                continue

        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4
                ip = addr.address
                if not ip.startswith("127."):
                    ip_info.append(
                        (
                            ip,
                            {
                                "interface": interface,
                                "type": "ipv4",
                                "speed": getattr(stats, "speed", 0),
                                "is_private": ipaddress.ip_address(ip).is_private,
                            },
                        )
                    )
            elif addr.family == socket.AF_INET6:  # IPv6
                # 过滤本地链接地址和回环地址
                ip = addr.address.split("%")[0]  # 移除接口标识符
                if not ip.startswith("fe80:") and not ip.startswith("::1"):
                    ip_info.append(
                        (
                            ip,
                            {
                                "interface": interface,
                                "type": "ipv6",
                                "speed": getattr(stats, "speed", 0),
                                "is_private": ipaddress.ip_address(ip).is_private,
                            },
                        )
                    )

    scored_ips = [(ip_info, score_ip(ip_info)) for ip_info in ip_info]
    scored_ips.sort(key=lambda x: x[1], reverse=True)
    return [ip_info[0] for ip_info, _ in scored_ips[:3]]


def tcp_ping(ip: str, port: int, timeout=3) -> Tuple[Optional[float], bool]:
    try:
        start_time = time.time()
        if is_ipv6(ip):
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            ipv6_flag = True
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ipv6_flag = False

        sock.settimeout(timeout)
        sock.connect((ip, port))
        end_time = time.time()
        delay = (end_time - start_time) * 1000
        sock.close()
        return delay, ipv6_flag
    except (socket.timeout, socket.error):
        return None, False


def tcp_ping_test(ip_list: List[str], port: int, timeout=5, count=3) -> Tuple[Optional[str], Optional[float]]:
    min_delay = None
    min_ip = None

    for ip in ip_list:
        delays = []

        for _ in range(count):
            delay, ipv6_flag = tcp_ping(ip, port, timeout)
            if delay is not None:
                delays.append(delay)

        if delays and all(delay is not None for delay in delays):  # 确保三次都有延迟值
            min_delay_ip = min(delays)
            if min_delay is None or min_delay_ip < min_delay:
                min_delay = min_delay_ip
                min_ip = ip if not ipv6_flag else f"[{ip}]"

    return min_ip, min_delay


def is_ipv6(ip: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except socket.error:
        return False
