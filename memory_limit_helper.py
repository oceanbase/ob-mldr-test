#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
操作系统级别的内存限制工具
使用 cgroups v2 来限制 observer 进程的物理内存占用
"""

import os
import subprocess
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryLimitHelper:
    """使用 cgroups 限制进程内存的工具类"""
    
    def __init__(self, cgroup_name: str = 'observer_limit', memory_limit_mb: int = 16384):
        """
        初始化内存限制工具
        
        Args:
            cgroup_name: cgroup 名称，默认 'observer_limit'
            memory_limit_mb: 内存限制（MB），默认 16GB (16384 MB)
        """
        self.cgroup_name = cgroup_name
        self.memory_limit_mb = memory_limit_mb
        # 初始路径，会在 _check_cgroup_v2 中根据实际挂载点更新
        self.cgroup_path = f'/sys/fs/cgroup/{cgroup_name}'
        self._is_v2 = True  # 默认为 v2，会在 _check_cgroup_v2 中更新
        
    def _find_cgroup_mount_point(self) -> tuple[Optional[str], bool]:
        """
        查找 cgroup 挂载点，支持 v1 和 v2
        
        Returns:
            tuple[Optional[str], bool]: (挂载点路径, 是否为 v2)
        """
        try:
            # 方法1: 检查 cgroup v2
            with open('/proc/self/mountinfo', 'r') as f:
                for line in f:
                    if 'cgroup2' in line:
                        # 解析挂载点（第5个字段）
                        parts = line.split()
                        if len(parts) >= 5:
                            mount_point = parts[4]
                            if os.path.exists(mount_point) and os.path.isdir(mount_point):
                                # 检查是否有 memory.max（v2 的特征）
                                #memory_file = os.path.join(mount_point, 'memory.max')
                                #if os.path.exists(memory_file):
                                return mount_point, True
            
            # 方法2: 检查 cgroup v1 memory 控制器
            memory_cgroup_path = '/sys/fs/cgroup/memory'
            if os.path.exists(memory_cgroup_path):
                # 检查是否有 memory.limit_in_bytes（v1 的特征）
                limit_file = os.path.join(memory_cgroup_path, 'memory.limit_in_bytes')
                if os.path.exists(limit_file):
                    return memory_cgroup_path, False
            
            return None, False
        except Exception as e:
            logger.warning(f"Failed to find cgroup mount point: {e}")
            return None, False
    
    def _check_cgroup_v2(self) -> bool:
        """检查系统是否支持 cgroups v2，如果不支持则尝试使用 v1"""
        mount_point, is_v2 = self._find_cgroup_mount_point()
        if mount_point:
            if is_v2:
                logger.info(f"Found cgroup v2 mount point: {mount_point}")
                self.cgroup_path = os.path.join(mount_point, self.cgroup_name)
                self._is_v2 = True
            else:
                logger.info(f"Found cgroup v1 memory controller: {mount_point}")
                self.cgroup_path = os.path.join(mount_point, self.cgroup_name)
                self._is_v2 = False
            return True
        return False
    
    def setup_memory_limit(self) -> bool:
        """
        设置内存限制 cgroup（统一使用 sudo）
        
        Returns:
            bool: 是否成功
        """
        if not self._check_cgroup_v2():
            logger.warning("cgroups not available, cannot set memory limit")
            return False
        
        try:
            memory_limit_bytes = self.memory_limit_mb * 1024 * 1024
            
            # 如果目录已存在，先清理旧的进程和目录，避免影响新的测试
            if os.path.exists(self.cgroup_path):
                logger.info(f"Cgroup directory already exists: {self.cgroup_path}, cleaning up for new test")
                self.cleanup()
            
            # 创建 cgroup 目录（无论之前是否存在，cleanup 后都需要重新创建）
            result = subprocess.run(
                ['sudo', 'mkdir', '-p', self.cgroup_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to create cgroup directory: {result.stderr}")
                return False
            logger.info(f"Created cgroup directory: {self.cgroup_path}")
            
            # 根据 cgroup 版本使用不同的文件
            if self._is_v2:
                memory_limit_file = os.path.join(self.cgroup_path, 'memory.max')
                memory_swap_max_file = os.path.join(self.cgroup_path, 'memory.swap.max')
                memory_oom_file = os.path.join(self.cgroup_path, 'memory.oom.group')
                
                # 设置内存限制
                result = subprocess.run(
                    ['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {memory_limit_file}'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set memory limit: {result.stderr}")
                    return False
                
                # 设置 swap 限制为 0，禁止使用 swap
                if os.path.exists(memory_swap_max_file):
                    result = subprocess.run(
                        ['sudo', 'sh', '-c', 'echo 0 > ' + memory_swap_max_file],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.returncode != 0:
                        logger.warning(f"Failed to disable swap: {result.stderr.strip()}")
                        logger.warning("Swap may still be used. Process may not trigger OOM when memory limit is exceeded.")
                    else:
                        logger.info("Disabled swap (memory.swap.max = 0) to ensure OOM when memory limit is exceeded")
                else:
                    logger.warning(f"memory.swap.max not found, swap may still be used")
                
                logger.info(f"Set memory limit to {self.memory_limit_mb} MB ({memory_limit_bytes} bytes) using cgroup v2 (swap disabled)")
            else:
                # cgroup v1: 需要同时设置 memory.memsw.limit_in_bytes 和 memory.limit_in_bytes 来禁止 swap
                memory_limit_file = os.path.join(self.cgroup_path, 'memory.limit_in_bytes')
                memory_memsw_limit_file = os.path.join(self.cgroup_path, 'memory.memsw.limit_in_bytes')
                memory_oom_file = os.path.join(self.cgroup_path, 'memory.oom_control')
                
                swap_disabled = False
                
                # 重要：必须先设置 memory.memsw.limit_in_bytes（内存+swap 总和限制）
                # 设置为与内存限制相同，这样就不会使用 swap
                if os.path.exists(memory_memsw_limit_file):
                    # 先设置 memory.memsw.limit_in_bytes（必须先设置这个）
                    result = subprocess.run(
                        ['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {memory_memsw_limit_file}'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if result.returncode != 0:
                        # 如果设置失败，可能是 swap accounting 未启用或需要先设置父 cgroup
                        logger.warning(f"Failed to set memory+swap limit: {result.stderr.strip()}")
                        logger.warning("Swap accounting may not be enabled or requires parent cgroup configuration.")
                        logger.warning("Will set memory limit only (swap may still be used)")
                    else:
                        swap_disabled = True
                        logger.info(f"Set memory+swap limit to {self.memory_limit_mb} MB ({memory_limit_bytes} bytes) to disable swap")
                else:
                    logger.warning(f"memory.memsw.limit_in_bytes not found, swap may still be used")
                
                # 再设置 memory.limit_in_bytes（纯内存限制）
                result = subprocess.run(
                    ['sudo', 'sh', '-c', f'echo {memory_limit_bytes} > {memory_limit_file}'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set memory limit: {result.stderr}")
                    return False
                
                if swap_disabled:
                    logger.info(f"Set memory limit to {self.memory_limit_mb} MB ({memory_limit_bytes} bytes) using cgroup v1 (swap disabled)")
                else:
                    logger.info(f"Set memory limit to {self.memory_limit_mb} MB ({memory_limit_bytes} bytes) using cgroup v1 (swap may still be used)")
            
            # 启用内存回收（OOM killer）
            if os.path.exists(memory_oom_file):
                if self._is_v2:
                    subprocess.run(
                        ['sudo', 'sh', '-c', f'echo 1 > {memory_oom_file}'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    logger.info("Enabled OOM group kill (cgroup v2)")
                else:
                    # v1: 在 v1 中，memory.oom_control 通常是只读的
                    # OOM kill 默认启用，这里只记录日志
                    logger.info("OOM kill should be enabled by default (cgroup v1)")
            
            return True
        except PermissionError:
            logger.error("Permission denied: need root privileges to set cgroup limits")
            return False
        except OSError as e:
            if e.errno == 30:  # Read-only file system
                logger.error(f"Cannot set memory limit: cgroup filesystem is read-only. "
                            f"Please check if cgroup v2 is properly mounted. Error: {e}")
            else:
                logger.error(f"Failed to setup memory limit: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to setup memory limit: {e}")
            return False
    
    def add_process_to_cgroup(self, pid: int) -> bool:
        """
        将进程添加到 cgroup（统一使用 sudo）
        
        Args:
            pid: 进程 ID
            
        Returns:
            bool: 是否成功
        """
        try:
            cgroup_procs_file = os.path.join(self.cgroup_path, 'cgroup.procs')
            if not os.path.exists(cgroup_procs_file):
                logger.error(f"cgroup.procs file not found: {cgroup_procs_file}")
                return False
            
            result = subprocess.run(
                ['sudo', 'sh', '-c', f'echo {pid} > {cgroup_procs_file}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to add process to cgroup: {result.stderr}")
                return False
            
            logger.info(f"Added process {pid} to cgroup {self.cgroup_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add process to cgroup: {e}")
            return False
    
    def find_observer_pids(self) -> list:
        """
        查找所有 observer 进程的 PID
        
        Returns:
            list: PID 列表
        """
        pids = []
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'observer'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]
        except Exception as e:
            logger.warning(f"Failed to find observer PIDs: {e}")
        return pids
    
    def apply_limit_to_observer(self) -> bool:
        """
        查找 observer 进程并应用内存限制（统一使用 sudo）
        
        Returns:
            bool: 是否成功
        """
        if not self.setup_memory_limit():
            return False
        
        # 等待一下，确保 observer 已经启动
        time.sleep(2)
        
        pids = self.find_observer_pids()
        if not pids:
            logger.warning("No observer processes found")
            return False
        
        success = True
        for pid in pids:
            if not self.add_process_to_cgroup(pid):
                success = False
        
        return success
    
    def check_oom(self) -> tuple[bool, Optional[str]]:
        """
        检查 cgroup 是否发生了 OOM（需要 root 权限）
        
        Returns:
            tuple[bool, Optional[str]]: (是否发生 OOM, OOM 详细信息)
        """
        try:
            if self._is_v2:
                # cgroup v2: 检查 memory.events 文件
                memory_events_file = os.path.join(self.cgroup_path, 'memory.events')
                if not os.path.exists(memory_events_file):
                    return False, None
                
                oom_count = 0
                with open(memory_events_file, 'r') as f:
                    for line in f:
                        if line.startswith('oom_kill '):
                            oom_count = int(line.split()[1])
                            break  # 只需检查 oom_kill（实际 kill 次数）
                
                if oom_count > 0:
                    # 获取内存限制（用于提示）
                    memory_max_file = os.path.join(self.cgroup_path, 'memory.max')
                    max_mb = "unlimited"
                    
                    if os.path.exists(memory_max_file):
                        with open(memory_max_file, 'r') as f:
                            max_bytes = f.read().strip()
                            if max_bytes != 'max':
                                max_mb = int(max_bytes) // (1024 * 1024)
                    
                    oom_info = f"\nObserver process was killed due to OOM. Memory limit: {max_mb} MB (cgroup v2)"
                    return True, oom_info

                return False, None

            else:
                # cgroup v1: 检查 memory.oom_control 文件
                memory_oom_control_file = os.path.join(self.cgroup_path, 'memory.oom_control')
                if not os.path.exists(memory_oom_control_file):
                    return False, None
                
                with open(memory_oom_control_file, 'r') as f:
                    oom_control_content = f.read()
                
                # 在 v1 中，OOM 信息在 memory.oom_control 中
                # 格式类似: oom_kill_disable 0\nunder_oom 0\noom_kill 1\n...
                # 检查 oom_kill 值，如果为 1 表示发生了 OOM kill
                oom_kill = 0
                under_oom = False
                for line in oom_control_content.strip().split('\n'):
                    if line.startswith('oom_kill '):
                        oom_kill = int(line.split()[1])
                    elif line.startswith('under_oom '):
                        under_oom = int(line.split()[1]) > 0
                
                # 如果 oom_kill > 0，说明发生了 OOM kill
                if oom_kill > 0:
                    # 获取内存使用情况
                    memory_limit_file = os.path.join(self.cgroup_path, 'memory.limit_in_bytes')

                    max_mb = 0
                    
                    if os.path.exists(memory_limit_file):
                        with open(memory_limit_file, 'r') as f:
                            max_bytes = f.read().strip()
                            if max_bytes != '9223372036854771712':  # v1 的 "unlimited" 值
                                max_mb = int(max_bytes) // (1024 * 1024)
                    
                    oom_info = f"\nObserver process was killed due to OOM (Out of Memory). Memory limit: {max_mb} MB, OOM kill count: {oom_kill} (cgroup v1)"
                    return True, oom_info
                
                # 如果 under_oom 为 1，表示当前处于 OOM 状态（但可能还没被 kill）
                if under_oom:
                    # 获取内存使用情况
                    memory_usage_file = os.path.join(self.cgroup_path, 'memory.usage_in_bytes')
                    memory_limit_file = os.path.join(self.cgroup_path, 'memory.limit_in_bytes')
                    
                    current_mb = 0
                    max_mb = 0
                    
                    if os.path.exists(memory_usage_file):
                        with open(memory_usage_file, 'r') as f:
                            current_mb = int(f.read().strip()) // (1024 * 1024)
                    
                    if os.path.exists(memory_limit_file):
                        with open(memory_limit_file, 'r') as f:
                            max_bytes = f.read().strip()
                            if max_bytes != '9223372036854771712':  # v1 的 "unlimited" 值
                                max_mb = int(max_bytes) // (1024 * 1024)
                    
                    oom_info = f"Observer process OOM condition detected (under_oom=1). Memory limit: {max_mb} MB, Current usage: {current_mb} MB (cgroup v1)"
                    return True, oom_info
            
            return False, None
        except Exception as e:
            logger.warning(f"Failed to check OOM: {e}")
            return False, None
    
    def cleanup(self):
        """
        清理 cgroup：删除目录
        这很重要：如果历史峰值统计没有被重置，可能会在检查 OOM 时误判
        例如：上一个测试中 observer 达到了很高的内存峰值，如果新测试中 observer 因其他原因退出，
        检查 OOM 时看到历史峰值很高，可能会误判为内存超限导致的 OOM
        通过删除并重新创建 cgroup，可以确保统计是干净的
        """
        try:
            if not os.path.exists(self.cgroup_path):
                return
            
            # 先尝试直接删除目录（如果进程已经退出，可以直接删除，使用 sudo）
            result = subprocess.run(
                ['sudo', 'rmdir', self.cgroup_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Cleaned up cgroup: {self.cgroup_path}")
                return
            # 如果删除失败，说明 cgroup 中还有进程，需要先移动进程
            
            # 如果直接删除失败，说明还有进程在 cgroup 中，需要先移动
            cgroup_procs_file = os.path.join(self.cgroup_path, 'cgroup.procs')
            if os.path.exists(cgroup_procs_file):
                # 读取当前 cgroup 中的所有进程
                with open(cgroup_procs_file, 'r') as f:
                    pids = [pid.strip() for pid in f.read().strip().split('\n') if pid.strip()]
                
                if pids:
                    logger.info(f"Found {len(pids)} process(es) in cgroup, moving them to root cgroup before deletion")
                    
                    # 确定根 cgroup 的路径
                    if self._is_v2:
                        root_cgroup_procs = '/sys/fs/cgroup/cgroup.procs'
                    else:
                        root_cgroup_procs = '/sys/fs/cgroup/memory/cgroup.procs'
                    
                    # 将每个进程移到根 cgroup（使用 sudo）
                    moved_count = 0
                    for pid in pids:
                        try:
                            # 验证进程是否还存在
                            if not os.path.exists(f'/proc/{pid}'):
                                continue
                            
                            # 将进程移到根 cgroup
                            result = subprocess.run(
                                ['sudo', 'sh', '-c', f'echo {pid} > {root_cgroup_procs}'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            if result.returncode == 0:
                                moved_count += 1
                            else:
                                logger.debug(f"Failed to move process {pid} to root cgroup: {result.stderr}")
                        except Exception as e:
                            # 进程可能已经退出，或者没有权限移动
                            logger.debug(f"Failed to move process {pid} to root cgroup: {e}")
                    
                    if moved_count > 0:
                        logger.info(f"Moved {moved_count} process(es) from cgroup {self.cgroup_name} to root cgroup")
            
            # 再次尝试删除目录（使用 sudo）
            result = subprocess.run(
                ['sudo', 'rmdir', self.cgroup_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                logger.info(f"Cleaned up cgroup: {self.cgroup_path}")
            else:
                logger.warning(f"Failed to delete cgroup directory after moving processes: {result.stderr}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cgroup: {e}")


def check_observer_oom(cgroup_name: str = 'observer_limit') -> tuple[bool, Optional[str]]:
    """
    检查 observer 进程是否发生了 OOM（需要 root 权限）
    
    Args:
        cgroup_name: cgroup 名称
    
    Returns:
        tuple[bool, Optional[str]]: (是否发生 OOM, OOM 详细信息)
    """
    helper = MemoryLimitHelper(cgroup_name=cgroup_name, memory_limit_mb=16384)  # memory_limit 不影响检查
    return helper.check_oom()


def set_observer_memory_limit(memory_limit_mb: int = 16384, cgroup_name: str = 'observer_limit') -> bool:
    """
    便捷函数：设置 observer 进程的内存限制（统一使用 sudo）
    
    Args:
        memory_limit_mb: 内存限制（MB），默认 16GB
        cgroup_name: cgroup 名称
        
    Returns:
        bool: 是否成功
    """
    helper = MemoryLimitHelper(cgroup_name=cgroup_name, memory_limit_mb=memory_limit_mb)
    return helper.apply_limit_to_observer()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python memory_limit_helper.py <memory_limit_mb> [cgroup_name]")
        print("Example: python memory_limit_helper.py 16384 observer_limit")
        sys.exit(1)
    
    memory_limit_mb = int(sys.argv[1])
    cgroup_name = sys.argv[2] if len(sys.argv) > 2 else 'observer_limit'
    
    logging.basicConfig(level=logging.INFO)
    success = set_observer_memory_limit(memory_limit_mb, cgroup_name)
    
    if success:
        print(f"Successfully set memory limit to {memory_limit_mb} MB")
    else:
        print("Failed to set memory limit")
        sys.exit(1)

