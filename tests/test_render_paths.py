"""M7e CPU-only tests for the PURE sendcmd crop-path generator
(``clipper.render_paths.crop_cmd_from_path``) — no I/O, no ffmpeg,
no GPU (docs/spec/m7.md M7e, card test list).
"""
import re

import pytest

from clipper.models import CropKey
from clipper.render_paths import CROP_CMD_FILENAME, crop_cmd_from_path

FPS = 30.0
LINE_RE = re.compile(r"-?\d+(?:\.\d+)? crop x -?\d+, crop y -?\d+;$")


def key(t: float, x: int, y: int = 0, w: int = 608, h: int = 1080) -> CropKey:
    return CropKey(t=t, x=x, y=y, w=w, h=h)


def test_filename_is_crop_cmd():
    assert CROP_CMD_FILENAME == "crop.cmd"


def test_t0_initial_command_present_as_first_line():
    out = crop_cmd_from_path([key(0.0, 656), key(1 / FPS, 660)], FPS)
    lines = out.splitlines()
    assert lines[0] == "0 crop x 656, crop y 0;"


def test_dense_keys_produce_one_line_each_in_order():
    keys = [key(i / FPS, 500 + i, y=0) for i in range(121)]  # dense 4s grid
    out = crop_cmd_from_path(keys, FPS)
    lines = out.splitlines()
    assert len(lines) == 121
    assert lines[0] == "0 crop x 500, crop y 0;"
    assert lines[1] == f"{1 / FPS:g} crop x 501, crop y 0;"
    assert lines[120] == f"{120 / FPS:g} crop x 620, crop y 0;"


def test_every_line_matches_sendcmd_crop_syntax():
    keys = [key(i / FPS, 720 - 5 * i, y=10 * i) for i in range(7)]
    out = crop_cmd_from_path(keys, FPS)
    lines = out.splitlines()
    assert lines, "expected at least the t=0 line"
    for line in lines:
        assert LINE_RE.fullmatch(line), line


def test_output_is_deterministic_and_ends_with_newline():
    keys = [key(i / FPS, 100 + i) for i in range(5)]
    a = crop_cmd_from_path(keys, FPS)
    b = crop_cmd_from_path(
        [CropKey(t=k.t, x=k.x, y=k.y, w=k.w, h=k.h) for k in keys], FPS)
    assert a == b  # byte-identical
    assert a.endswith("\n")


def test_single_key_is_valid_t0_file():
    out = crop_cmd_from_path([key(0.0, 33, y=4, w=9, h=16)], FPS)
    assert out == "0 crop x 33, crop y 4;\n"


def test_fps_must_be_positive():
    for bad in (0.0, -30.0):
        with pytest.raises(ValueError, match="fps"):
            crop_cmd_from_path([key(0.0, 5)], bad)


def test_empty_crop_path_rejected():
    with pytest.raises(ValueError, match="t=0|empty"):
        crop_cmd_from_path([], FPS)


def test_first_key_must_be_t0():
    with pytest.raises(ValueError, match="t=0"):
        crop_cmd_from_path([key(1 / FPS, 5), key(2 / FPS, 6)], FPS)
