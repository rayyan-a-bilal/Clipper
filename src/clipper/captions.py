"""M7a — pure caption ASS builder (see `docs/spec/m7.md`, M7a decisions).

PURE module: imports only `clipper.config` + `clipper.models`. No I/O, no
ffmpeg, no numpy, no network. Deterministic — same inputs always produce
byte-identical ASS output.

Public API:
    words_to_ass(words, plan, cfg) -> str
"""

MIN_DISPLAY_S = 0.6
HIGHLIGHT_COLOR = "&H00FFFF&"  # yellow (ASS BGR hex) — unsung karaoke
SUNG_COLOR = "&HFFFFFF&"  # white — sung karaoke
HOOK_FONT_SIZE = 72  # separate top style (judgment note, docs/spec/m7.md)

_SCRIPT_INFO = (
    "[Script Info]\n"
    "ScriptType: v4.00+\n"
    "PlayResX: {px}\n"
    "PlayResY: {py}\n"
    "WrapStyle: 0\n"
)
_V4_STYLES_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
    "MarginL, MarginR, MarginV, Encoding"
)
_EVENTS_FORMAT = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text"
)


def _fmt_time(seconds: float) -> str:
    """ASS timestamp `H:MM:SS.cc` (centiseconds) from non-negative seconds."""
    cs = int(round(max(0.0, seconds) * 100.0))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    c = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _parse_time(ts: str) -> float:
    """Inverse of `_fmt_time` (used by tests)."""
    h, rest = ts.split(":", 1)
    m, rest = rest.split(":", 1)
    s, c = rest.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(s) + int(c) / 100.0)


def _select_words(
    words: list, plan
) -> list[tuple[float, float, str]]:
    """Clip-relative (start, end) intervals for words overlapping the plan.

    A word overlapping `[plan.start, plan.end]` is kept; relative times are
    `max(0.0, w.* - plan.start)` (a word reaching in from before the cut is
    clamped to t >= 0 — judgment note; never a negative ASS timestamp).
    """
    out: list[tuple[float, float, str]] = []
    for w in words:
        if w.end > plan.start and w.start < plan.end:
            rs = max(0.0, w.start - plan.start)
            re_ = max(rs, w.end - plan.start)
            out.append((rs, re_, w.text))
    return out


def _group_words(
    words: list[tuple[float, float, str]], max_words: int, max_chars: int,
    max_lines: int
) -> list[list[list[tuple[float, float, str]]]]:
    """Greedy deterministic grouping. Returns a list of groups; each group is
    a list of LINES (each a list of words). Invariants: <= max_words words per
    group, <= max_chars chars per line, <= max_lines lines per group — so NO
    word is ever dropped. A word longer than one line takes a line to itself
    (words are the unit; never split)."""
    groups: list[list[list[tuple[float, float, str]]]] = []
    lines: list[list[tuple[float, float, str]]] = []
    line: list[tuple[float, float, str]] = []
    line_chars = 0
    group_count = 0

    def flush() -> None:
        nonlocal lines, line, line_chars, group_count
        if any(lines):
            groups.append([ln for ln in lines if ln])
        lines, line, line_chars, group_count = [], [], 0, 0

    for item in words:
        n = len(item[2])
        # 1. join the current line (fits char-wise, group not full)
        if (
            line
            and group_count < max_words
            and line_chars + 1 + n <= max_chars
        ):
            line.append(item)
            line_chars += 1 + n
            group_count += 1
            continue
        # 2. open a NEW LINE within the current group (slot left + line cap)
        if lines and len(lines) < max_lines and group_count < max_words:
            lines.append([item])
            line = lines[-1]
            line_chars = n
            group_count += 1
            continue
        # 3. flush the group; this word starts a new group / first line
        flush()
        lines = [[item]]
        line = lines[0]
        line_chars = n
        group_count = 1
    flush()
    return groups


def _word_karaoke(rs: float, re_: float, text: str) -> str:
    cs = max(1, int(round((re_ - rs) * 100.0)))
    return (
        rf"{{\kf{cs}}}{{\c{HIGHLIGHT_COLOR}}}{text}"
        rf"{{\r}}{{\c{SUNG_COLOR}}}{text}"
    )


def _group_text(lines: list[list[tuple[float, float, str]]]) -> str:
    """Render one group (pre-packed lines) as karaoke ASS text."""
    return "\\N".join(
        " ".join(_word_karaoke(rs, re_, text) for rs, re_, text in line)
        for line in lines
    )


def _style_line(
    name: str, font: str, size: int, alignment: int, margin_l: int,
    margin_r: int, margin_v: int
) -> str:
    return (
        f"Style: {name},{font},{size},"
        f"&H00FFFFFF&,&H000000FF&,&H80000000&,&H00000000&,"
        f"1,0,0,0,100,100,0,0,1,4,0,{alignment},{margin_l},{margin_r},"
        f"{margin_v},1"
    )


def words_to_ass(words: list, plan, cfg) -> str:
    """Build the deterministic ASS (caption + hook overlay) document for one
    clip from its transcript words, draft ClipPlan and config.

    - words overlapping `[plan.start, plan.end]` only, clip-relative times
    - <= `cfg.captions.max_words_per_group` per group,
      <= `cfg.captions.max_chars_per_line` per line, <= `max_lines` lines
    - min display `MIN_DISPLAY_S` (0.6s) per caption event
    - per-word `\\kf` karaoke (unsung yellow `&H00FFFF&` -> sung white
      `&HFFFFFF&`)
    - hook overlay: separate top style, first
      `min(cfg.clips.hook_overlay_duration, clip duration)` seconds, only
      when `plan.hook_overlay` is non-empty
    - empty `words` -> valid ASS with ZERO caption events
    """
    cap = cfg.captions
    selected = _select_words(words, plan)

    groups = _group_words(
        selected, cap.max_words_per_group, cap.max_chars_per_line,
        cap.max_lines
    )

    events: list[str] = []
    duration = plan.end - plan.start
    hook_text = " ".join((plan.hook_overlay or "").split())
    if hook_text:
        hook_end = min(cfg.clips.hook_overlay_duration, max(0.0, duration))
        if hook_end > 0.0:
            events.append(
                f"Dialogue: 0,{_fmt_time(0.0)},{_fmt_time(hook_end)},"
                f"Hook,,0,0,0,,{hook_text}"
            )

    for i, group in enumerate(groups):
        start = group[0][0][0]
        if i + 1 < len(groups):
            end = groups[i + 1][0][0][0]
        else:
            end = group[-1][-1][1]
        end = max(end, start + MIN_DISPLAY_S)
        text = _group_text(group)
        events.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},"
            f"Caption,,0,0,0,,{text}"
        )

    caption_style = _style_line(
        "Caption", cap.font, cap.font_size, alignment=2,
        margin_l=cap.safe_zone_right, margin_r=cap.safe_zone_right,
        margin_v=cap.margin_v
    )
    hook_style = _style_line(
        "Hook", cap.font, HOOK_FONT_SIZE, alignment=8,
        margin_l=cap.safe_zone_right, margin_r=cap.safe_zone_right,
        margin_v=cap.safe_zone_top
    )

    parts = [
        _SCRIPT_INFO.format(px=cfg.render.output_width,
                            py=cfg.render.output_height),
        "\n",
        "[V4+ Styles]\n",
        _V4_STYLES_FORMAT + "\n",
        caption_style + "\n",
        hook_style + "\n",
        "\n",
        "[Events]\n",
        _EVENTS_FORMAT + "\n",
    ]
    for ev in events:
        parts.append(ev + "\n")
    return "".join(parts)
