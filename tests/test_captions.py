"""M7a — words_to_ass caption engine (CPU-only; no GPU/ffmpeg/network)."""

import re

import pytest

from clipper import captions
from clipper.config import CaptionsConfig, Config
from clipper.models import ClipPlan, Layout, Word


def W(start, end, text):
    return Word(start=start, end=end, text=text, prob=0.99)


def make_plan(start=10.0, end=22.0, hook="Watch this"):
    return ClipPlan(
        clip_id="j1-clip01",
        job_id="j1",
        start=start,
        end=end,
        layout=Layout.center_crop,
        crop_path=[],
        weighted_score=8.0,
        title="t",
        hook_overlay=hook,
        description="d",
        hashtags=["#x"],
        status="pending_qc",
    )


def cfg_default():
    return Config()


def events(ass_text):
    """Parse dialogue events into dicts (start/end as floats)."""
    out = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line[len("Dialogue:"):].strip().split(",", 9)
        out.append({
            "start": captions._parse_time(parts[1]),
            "end": captions._parse_time(parts[2]),
            "style": parts[3],
            "text": parts[9],
        })
    return out


def style_fields(ass_text, name):
    for line in ass_text.splitlines():
        if line.startswith(f"Style: {name},"):
            rest = line[len(f"Style: {name},"):].split(",")
            # font, size, prim, sec, outline_colour, back, bold, italic,
            # underline, strike, scaleX, scaleY, spacing, angle, border,
            # outline, shadow, alignment, margin_l, margin_r, margin_v, enc
            return {
                "font": rest[0], "size": rest[1],
                "primary": rest[2], "secondary": rest[3],
                "bold": rest[6],
                "border": rest[14], "outline": rest[15], "shadow": rest[16],
                "alignment": rest[17], "margin_l": rest[18],
                "margin_r": rest[19], "margin_v": rest[20],
            }
    raise AssertionError(f"style {name!r} not found")


def test_valid_ass_header_and_caption_style():
    out = captions.words_to_ass([W(10.0, 11.0, "hello")], make_plan(), cfg_default())
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out
    assert "ScriptType: v4.00+" in out
    assert "PlayResX: 1080" in out
    assert "PlayResY: 1920" in out
    s = style_fields(out, "Caption")
    assert s["font"] == "Montserrat ExtraBold"
    assert s["size"] == "88"
    assert s["bold"] == "1"
    assert s["outline"] == "4"
    assert s["shadow"] == "0"
    assert s["alignment"] == "2"
    # safe zones: right 150 (both margins), bottom MarginV 450
    assert s["margin_l"] == "150"
    assert s["margin_r"] == "150"
    assert s["margin_v"] == "450"


def test_event_lines_use_spec_conformant_dialogue_tag():
    # M7g regression: libass silently DROPS any event line not spelled
    # exactly `Dialogue:` (this was the reason real renders had zero
    # visible subtitles while captions.ass looked full of events).
    words = [W(10.5, 11.0, "hello"), W(12.0, 13.0, "world")]
    out = captions.words_to_ass(words, make_plan(), cfg_default())
    lines = out.splitlines()
    assert sum(1 for line in lines if line.startswith("Dialogue:")) >= 2
    assert not any(line.startswith("dialogue") for line in lines)


def test_hook_style_top_and_safe_zone():
    out = captions.words_to_ass([W(10.0, 11.0, "hello")], make_plan(), cfg_default())
    s = style_fields(out, "Hook")
    assert s["alignment"] == "8"       # top center
    assert s["size"] == "72"           # HOOK_FONT_SIZE
    assert s["margin_v"] == "220"      # top safe zone respected
    assert s["margin_r"] == "150"
    assert s["bold"] == "1"
    assert s["outline"] == "4"
    assert s["shadow"] == "0"


def test_clip_relative_times_multiple_groups():
    words = [
        W(10.5, 11.0, "aa"), W(11.5, 12.0, "bb"), W(12.5, 13.0, "cc"),
        W(13.5, 14.0, "dd"),  # group 1: 4 words -> flush at word cap
        W(15.0, 16.0, "ee"), W(17.0, 18.0, "ff"),  # group 2
    ]
    evs = [e for e in events(captions.words_to_ass(words, make_plan(), cfg_default()))
           if e["style"] == "Caption"]
    assert [e["start"] for e in evs] == [0.5, 5.0]  # clip-relative, not 10.5
    assert evs[1]["start"] == pytest.approx(15.0 - 10.0)


def test_word_overlapping_before_cut_clamped_to_zero():
    words = [W(9.0, 11.0, "hello"), W(12.0, 13.0, "hi")]
    out = captions.words_to_ass(words, make_plan(start=10.0), cfg_default())
    evs = [e for e in events(out) if e["style"] == "Caption"]
    assert evs[0]["start"] == 0.0  # clamped, never negative


def test_words_outside_clip_range_excluded():
    words = [W(0.0, 1.0, "early"), W(10.5, 11.0, "middle"), W(20.0, 21.0, "late")]
    out = captions.words_to_ass(words, make_plan(start=10.0, end=20.0), cfg_default())
    evs = [e for e in events(out) if e["style"] == "Caption"]
    text = " ".join(e["text"] for e in evs)
    assert "middle" in text
    assert "early" not in text
    assert "late" not in text


def test_grouping_word_cap_and_line_limits():
    words = [W(10.0 + i, 10.0 + i + 0.5, w)
             for i, w in enumerate(["one", "two", "three", "four",
                                    "five", "six", "seven", "eight"])]
    evs = [e for e in events(captions.words_to_ass(words, make_plan(), cfg_default()))
           if e["style"] == "Caption"]
    for e in evs:
        n_words = e["text"].count(r"\kf")
        assert 1 <= n_words <= 4
        lines = e["text"].split("\\N")
        assert len(lines) <= 2
        for line in lines:
            plain = re.sub(r"\{[^}]*\}", "", line).strip()
            toks = plain.split()  # M7h: each word appears ONCE
            assert len(" ".join(toks)) <= 20
            assert len(toks) <= 4
    # 8 distinct words all present
    for w in ["one", "two", "three", "four", "five", "six", "seven", "eight"]:
        assert w in out_text(evs)


def out_text(evs):
    return " ".join(e["text"] for e in evs)


def test_line_fill_forces_new_group_within_max_lines():
    long = "abcdefghij"  # 10 chars; two per line (with space = 21 > 20 -> one/line)
    words = [W(10.0 + i, 10.0 + i + 0.5, long) for i in range(6)]
    evs = [e for e in events(captions.words_to_ass(words, make_plan(), cfg_default()))
           if e["style"] == "Caption"]
    for e in evs:
        assert e["text"].count(r"\kf") <= 4
        lines = e["text"].split("\\N")
        assert len(lines) <= 2
        for line in lines:
            plain = re.sub(r"\{[^}]*\}", "", line).strip()
            toks = plain.split()
            assert len(toks) == 1  # 10-char word: one per line (21 > 20)
            # M7h: exact single copy (old format rendered `abcdefghijabcdefghij`)
            assert toks[0] == "abcdefghij"
    # NO word dropped: all 6 present (exactly once each — M7h)
    total = sum(e["text"].count("abcdefghij") for e in evs)
    assert total == 6


def test_karaoke_word_emitted_once_m7h():
    """M7h regression: each word is written EXACTLY once per event.

    The M7a format emitted the word twice (`{\\kf}{\\c..}w{\\r}{\\c..}w`);
    ASS format tags do not consume text, so real renders showed doubled
    words (`Oh,Oh, there'sthere's`). Karaoke colors come from the Caption
    STYLE: PrimaryColour = sung white, SecondaryColour = unsung yellow.
    """
    out = captions.words_to_ass(
        [W(10.0, 11.5, "Oh,"), W(11.6, 13.0, "there's")],
        make_plan(start=10.0, end=14.0), cfg_default()
    )
    evs = [e for e in events(out) if e["style"] == "Caption"]
    text = evs[0]["text"]
    # exactly one karaoke span + one copy per word — never `Oh,Oh,`
    assert text.count("Oh,") == 1
    assert text.count("there's") == 1
    assert text.count(r"\kf") == 2
    assert r"\c" not in text  # no inline color overrides at all
    assert r"{\kf150}Oh," in text  # 1.5s in centiseconds, word once
    s = style_fields(out, "Caption")
    assert s["primary"] == "&HFFFFFF&"   # sung = white (style, not inline)
    assert s["secondary"] == "&H00FFFF&"  # unsung highlight = yellow


def test_min_display_0_6_seconds():
    out = captions.words_to_ass(
        [W(10.0, 10.2, "hi")], make_plan(start=10.0, end=12.0), cfg_default()
    )
    evs = [e for e in events(out) if e["style"] == "Caption"]
    assert evs[0]["end"] - evs[0]["start"] == pytest.approx(0.6)


def test_hook_overlay_first_2_5s_only():
    evs = [e for e in events(captions.words_to_ass(
        [W(10.0, 11.0, "x")], make_plan(start=10.0, end=22.0,
                                        hook="Watch this"), cfg_default()))
        if e["style"] == "Hook"]
    assert len(evs) == 1
    assert evs[0]["start"] == 0.0
    assert evs[0]["end"] == pytest.approx(2.5)
    assert evs[0]["text"] == "Watch this"


def test_hook_overlay_capped_at_clip_duration():
    evs = [e for e in events(captions.words_to_ass(
        [W(10.0, 11.0, "x")], make_plan(start=10.0, end=11.5,
                                        hook="Watch this"), cfg_default()))
        if e["style"] == "Hook"]
    assert evs[0]["end"] == pytest.approx(1.5)


def test_hook_absent_when_empty():
    out = captions.words_to_ass(
        [W(10.0, 11.0, "x")], make_plan(hook=""), cfg_default()
    )
    assert not [e for e in events(out) if e["style"] == "Hook"]


def test_empty_words_valid_ass_no_caption_events():
    out = captions.words_to_ass([], make_plan(hook=""), cfg_default())
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out
    assert style_fields(out, "Caption")
    assert style_fields(out, "Hook")
    assert events(out) == []


def test_determinism_byte_identical():
    words = [W(10.0 + i, 10.0 + i + 0.4, f"w{i}") for i in range(7)]
    plan = make_plan()
    a = captions.words_to_ass(words, plan, cfg_default())
    b = captions.words_to_ass(list(words), plan, cfg_default())
    assert a == b


def test_cfg_drives_style_values():
    cfg = Config()
    cfg.captions = CaptionsConfig(
        font="Rounded", font_size=50, margin_v=100,
        safe_zone_right=60, safe_zone_top=90,
    )
    out = captions.words_to_ass(
        [W(10.0, 11.0, "x")], make_plan(hook="hi"), cfg
    )
    s = style_fields(out, "Caption")
    assert s["font"] == "Rounded"
    assert s["size"] == "50"
    assert s["margin_v"] == "100"
    assert s["margin_l"] == "60"
    assert s["margin_r"] == "60"
    h = style_fields(out, "Hook")
    assert h["margin_v"] == "90"
    assert h["margin_r"] == "60"
