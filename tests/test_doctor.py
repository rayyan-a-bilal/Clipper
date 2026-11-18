"""
Tests for the doctor command functionality.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

from clipper.cli import doctor


def test_doctor_command_runs():
    """Test that doctor command runs without error."""

    # Mock external dependencies to avoid actual system calls
    with patch('clipper.cli.get_config') as mock_config, \
         patch('clipper.cli.get_db') as mock_db, \
         patch('clipper.cli.get_devices') as mock_devices:

        # Setup mocks
        mock_config.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_db_instance.get_schema_version.return_value = 1
        mock_db.return_value = mock_db_instance

        mock_devices_instance = MagicMock()
        mock_devices_instance.get_all_devices.return_value = (
            {"nvidia": {}, "ffmpeg": {}}
        )
        mock_devices.return_value = mock_devices_instance

        # This should not raise any exception
        try:
            doctor()  # doctor command entry point
            assert True  # If we get here, no exception was raised
        except SystemExit as e:
            # Can be expected for argument parsing errors but shouldn't be other errors
            assert e.code == 0  # Success exit code

        # Test with missing essential components
        mock_devices_instance.get_all_devices.return_value = (
            {"nvidia": {"error": "not found"}, "ffmpeg": {"error": "not found"}}
        )

        try:
            doctor()
            assert True
        except SystemExit as e:
            assert e.code == 0


def test_doctor_with_temp_data_dir():
    """Test doctor with temporary data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['CLIPPER_DATA_DIR'] = temp_dir

        # Mock external dependencies to avoid actual system calls
        with patch('clipper.cli.get_config') as mock_config, \
             patch('clipper.cli.get_db') as mock_db, \
             patch('clipper.cli.get_devices') as mock_devices:

            mock_config.return_value = MagicMock()
            mock_db_instance = MagicMock()
            mock_db_instance.get_schema_version.return_value = 1
            mock_db.return_value = mock_db_instance

            mock_devices_instance = MagicMock()
            mock_devices_instance.get_all_devices.return_value = (
                {"nvidia": {}, "ffmpeg": {}}
            )
            mock_devices.return_value = mock_devices_instance

            mock_devices_instance = MagicMock()

            try:
                doctor()
                assert True
            except SystemExit as e:
                assert e.code == 0

        # Clean up
        del os.environ['CLIPPER_DATA_DIR']
