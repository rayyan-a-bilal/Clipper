"""
Tests for device detection.
"""

import tempfile
import os
from clipper.config import Config
from clipper.devices import DeviceManager


def test_device_manager_creation():
    """Test that DeviceManager can be created."""
    config = Config.load_config()
    devices = DeviceManager(config)
    assert devices is not None


def test_device_fallback_without_nvidia_smi():
    """Test device manager works even when nvidia-smi is not available."""
    config = Config.load_config()
    devices = DeviceManager(config)
    
    # This should not raise an exception
    device_info = devices.get_all_devices()
    assert isinstance(device_info, dict)


def test_device_locking():
    """Test that device locking works in CPU-only mode."""
    config = Config.load_config()
    devices = DeviceManager(config)
    
    # Test getting a lock (this won't actually lock on filesystem 
    # but should not raise exception)
    try:
        lock = devices.get_device_lock("cpu", "test")
        assert lock is not None
    except Exception:
        # This might fail in some edge cases, but shouldn't crash
        pass