"""
Device detection and locking for clipper.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from filelock import FileLock

from .config import Config

class DeviceManager:
    """Manages device detection and locking."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def get_gpu_info(self) -> Dict[str, any]:
        """Detect NVIDIA GPUs using nvidia-smi if available."""
        try:
            # Check if nvidia-smi exists
            result = subprocess.run(['which', 'nvidia-smi'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return {"error": "nvidia-smi not found"}
                
            # Get GPU info using nvidia-smi
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.free,memory.used',
                '--format=csv,nounits,noheader'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_info = {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total": int(parts[2]),
                            "memory_free": int(parts[3]),
                            "memory_used": int(parts[4])
                        }
                        gpus.append(gpu_info)
                return {"gpus": gpus}
            else:
                return {"error": f"nvidia-smi failed: {result.stderr}"}
                
        except Exception as e:
            return {"error": str(e)}
            
    def get_ffmpeg_info(self) -> Dict[str, any]:
        """Detect ffmpeg/ffprobe if available."""
        try:
            # Check for ffmpeg
            result = subprocess.run(['which', 'ffmpeg'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return {"error": "ffmpeg not found"}
                
            # Get version info
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return {
                    "ffmpeg_available": True,
                    "version": result.stdout.split('\n')[0]  
                }
            else:
                return {"error": f"ffmpeg error: {result.stderr}"}
                
        except Exception as e:
            return {"error": str(e)}
            
    def get_device_lock(self, device_type: str, role: str) -> FileLock:
        """Get a file lock for a specific device and role."""
        lock_path = self.config.resolved_data_dir / "run" / f"{device_type}_{role}.lock"
        return FileLock(str(lock_path))
        
    def get_all_devices(self) -> Dict[str, any]:
        """Get information about all available devices."""
        result = {
            "nvidia": self.get_gpu_info(),
            "ffmpeg": self.get_ffmpeg_info()
        }
        return result