"""Shared terminal presentation for mesh operator commands.

One styling dialect for the setup wizard, ``mesh fleet``, ``mesh status``
and the probe results, instead of each surface growing private helpers.

Every helper is TTY-gated: when stdout is not a terminal (agents, pipes,
CI, ``| cat``) the output degrades to the plain text these commands always
printed, so nothing that parses the current output ever breaks. ``NO_COLOR``
is honored for humans who want the plain form on a TTY.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import sys
import threading
from collections.abc import Sequence

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(text: str) -> int:
    """Length as a terminal renders it (ANSI escapes excluded)."""

    return len(_ANSI_RE.sub("", text))


def use_style(stream=None) -> bool:
    """Style only when a human is looking: TTY and NO_COLOR unset.

    ``CLICOLOR_FORCE``/``FORCE_COLOR`` (the ecosystem-standard overrides)
    style a non-TTY stream anyway: the manage board captures its child
    CLIs' output and re-prints it on a real terminal, and without the
    override every one of those children stripped its own styling.
    ``NO_COLOR`` still wins, per its spec.
    """

    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    # Per the conventions both variables come from, "0" means NOT forced
    # (FORCE_COLOR=0 is a common explicit-off idiom); any other non-empty
    # value forces.
    if any(
        os.environ.get(name, "") not in ("", "0")
        for name in ("CLICOLOR_FORCE", "FORCE_COLOR")
    ):
        return True
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def c(text: str, code: str, *, styled: bool | None = None) -> str:
    """ANSI-wrap ``text`` with ``code`` when styling is on; plain otherwise."""

    if styled is None:
        styled = use_style()
    return f"\033[{code}m{text}\033[0m" if styled else text


def bold(text: str, *, styled: bool | None = None) -> str:
    return c(text, "1", styled=styled)


def dim(text: str, *, styled: bool | None = None) -> str:
    return c(text, "2", styled=styled)


def green(text: str, *, styled: bool | None = None) -> str:
    return c(text, "32", styled=styled)


def red(text: str, *, styled: bool | None = None) -> str:
    return c(text, "31", styled=styled)


def yellow(text: str, *, styled: bool | None = None) -> str:
    return c(text, "33", styled=styled)


def cyan(text: str, *, styled: bool | None = None) -> str:
    return c(text, "36", styled=styled)


def ok(text: str, *, styled: bool | None = None) -> str:
    if styled is None:
        styled = use_style()
    mark = green("✓", styled=styled) if styled else "✓"
    return f"  {mark} {text}"


def warn(text: str, *, styled: bool | None = None) -> str:
    if styled is None:
        styled = use_style()
    mark = yellow("!", styled=styled) if styled else "!"
    return f"  {mark} {text}"


def fail(text: str, *, styled: bool | None = None) -> str:
    if styled is None:
        styled = use_style()
    mark = red("✗", styled=styled) if styled else "✗"
    return f"  {mark} {text}"


# The Verathos mark rendered FROM assets/logo.png (braille dots pack a
# 40x40 canvas into 10 rows, enough to resolve the concentric arcs;
# color sampled from the artwork). Regenerate with
# scripts/gen_terminal_logo.py whenever the asset changes.
_LOGO_ANSI = (
    '\x1b[38;2;62;137;255m⠀⠀⠀⣠⡴⠞⢛⣀⣀⣛⠳⢦⣄\x1b[0m',
    '\x1b[38;2;62;137;255m⠀⣠⡞⣩⡄⣀⣬⣭⣭⣭⣛⠷⣍⢳⣄\x1b[0m',
    '\x1b[38;2;62;137;255m⣰⠏⣼⠋⠀⠈⠀⠀⠀⠀⠙⠷⠙⣧⠹⣆\x1b[0m',
    '\x1b[38;2;62;137;255m⣿⢸⡇⣿⠀⠀⠀⠀⠀⠀⠀⠀⣀⠸⠇⣿\x1b[0m',
    '\x1b[38;2;62;137;255m⣿⢸⡇⣿⠀⠀⠀⠀⠀⠀⠀⠀⣿⢀⠀⣿\x1b[0m',
    '\x1b[38;2;62;137;255m⠘⠃⢻⣜⢷⣄⠀⠀⠀⠀⣠⡾⣣⡟⠘⠃\x1b[0m',
    '\x1b[38;2;62;137;255m⠀⠐⢶⣅⠀⢉⣀⣀⣘⣛⣭⠾⣫⡴⠂\x1b[0m',
    '\x1b[38;2;62;137;255m⠀⠀⠀⠙⠳⠶⣭⣭⣭⣭⠶⠞⠋\x1b[0m',
)
_LOGO_WIDTH = 16


def brand_banner(
    lines: Sequence[str], *, styled: bool | None = None
) -> list[str]:
    """The Verathos mark with product lines beside it, for welcome screens.

    Styled output pairs the braille-rendered logo with the text lines
    (first line bold, the rest dim), vertically centered. Plain (non-TTY,
    NO_COLOR) output degrades to the text lines alone: braille art in a
    pipe is noise to a parser and mojibake in a log.
    """

    if styled is None:
        styled = use_style()
    text_lines = [str(line) for line in lines]
    if not styled:
        return ["  " + line for line in text_lines]
    # Text sits high beside the mark (one row down at most), not
    # vertically centered: the product name should lead the banner.
    offset = min(1, max(0, len(_LOGO_ANSI) - len(text_lines)))
    out: list[str] = []
    for index, art in enumerate(_LOGO_ANSI):
        pad = " " * (_LOGO_WIDTH - visible_len(art))
        text_index = index - offset
        text = ""
        if 0 <= text_index < len(text_lines):
            text = "   " + (
                bold(text_lines[text_index], styled=True)
                if text_index == 0
                else dim(text_lines[text_index], styled=True)
            )
        out.append((art + pad + text).rstrip())
    for extra_index in range(len(_LOGO_ANSI) - offset, len(text_lines)):
        out.append(
            " " * (_LOGO_WIDTH + 3)
            + dim(text_lines[extra_index], styled=True)
        )
    return out


def box(title: str, subtitle: str = "", *, styled: bool | None = None) -> str:
    """Banner box for wizard/summary headers."""

    if styled is None:
        styled = use_style()
    width = max(len(title), len(subtitle)) + 4

    def row(text: str) -> str:
        return "│ " + text.ljust(width - 4) + " │"

    lines = ["╭" + "─" * (width - 2) + "╮", row(title)]
    if subtitle:
        lines.append(row(subtitle))
    lines.append("╰" + "─" * (width - 2) + "╯")
    return c("\n".join(lines), "36", styled=styled)


def step(index: int, total: int, title: str, *, styled: bool | None = None) -> str:
    if styled is None:
        styled = use_style()
    return "\n" + c(f"▸ Step {index}/{total} · {title}", "1;36", styled=styled) + "\n"


def section(title: str, *, styled: bool | None = None) -> str:
    if styled is None:
        styled = use_style()
    return bold(title, styled=styled)


def badge(status: str, *, styled: bool | None = None) -> str:
    """Colorize a well-known status word; pass through anything else."""

    if styled is None:
        styled = use_style()
    if not styled:
        return status
    # Match on the word, color the string as given (it may be pre-padded so
    # alignment math ran before any ANSI escapes were added).
    lowered = status.strip().lower()
    if lowered in ("serving", "online", "ready", "verified", "ok"):
        return green(status, styled=True)
    if lowered in ("launching", "joining", "driving", "loading", "stale"):
        return yellow(status, styled=True)
    if lowered in ("error", "failed", "dead", "stopped"):
        return red(status, styled=True)
    return status


class _SpinnerHandle:
    """Live label updates for a running spinner (or its plain fallback)."""

    def __init__(self, state: dict, *, plain: bool, stream) -> None:
        self._state = state
        self._plain = plain
        self._stream = stream

    def update(self, label: str) -> None:
        if label == self._state["label"]:
            return
        self._state["label"] = label
        if self._plain:
            # Non-TTY consumers get one line per state CHANGE, never an
            # animation; the output stays grep/parse friendly.
            print(f"  {label}", file=self._stream, flush=True)


@contextlib.contextmanager
def spinner(label: str, *, stream=None):
    """Animated wait indicator, same braille glyphs as the vLLM wizard.

    TTY-gated like every helper here: on a terminal a single line spins
    and rewrites in place (cleared on exit); piped/agent output prints the
    label once plus one line per ``update()`` change. The yielded handle's
    ``update(text)`` feeds live progress (e.g. a worker's "fetching 45%")
    into the same line.
    """

    stream = stream if stream is not None else sys.stdout
    state = {"label": label}
    if not use_style(stream):
        print(f"  {label}", file=stream, flush=True)
        yield _SpinnerHandle(state, plain=True, stream=stream)
        return
    done = threading.Event()

    def _spin() -> None:
        for glyph in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if done.is_set():
                return
            stream.write(f"\r  \033[36m{glyph}\033[0m {state['label']}\033[K")
            stream.flush()
            done.wait(0.1)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield _SpinnerHandle(state, plain=False, stream=stream)
    finally:
        done.set()
        thread.join(timeout=0.5)
        stream.write("\r\033[K")
        stream.flush()


def table(
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = (),
    indent: str = "  ",
    align: str = "",
    styled: bool | None = None,
) -> list[str]:
    """Aligned columns; header dimmed when styled.

    Widths and padding use the VISIBLE cell length (ANSI escapes
    stripped), so pre-colored cells align exactly like plain ones and
    plain tables render byte-identically to before. ``align`` marks
    columns right-aligned by position ("r" per column, e.g. "lrr..";
    anything but "r" keeps the default left alignment).
    """

    if styled is None:
        styled = use_style()
    all_rows = ([list(header)] if header else []) + [list(row) for row in rows]
    if not all_rows:
        return []
    columns = max(len(row) for row in all_rows)
    widths = [0] * columns
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], visible_len(str(cell)))

    def pad(cell: str, i: int) -> str:
        gap = widths[i] - visible_len(cell)
        if gap <= 0:
            return cell
        if i < len(align) and align[i] == "r":
            return " " * gap + cell
        return cell + " " * gap

    def fmt(row: Sequence[str], *, dim_row: bool = False) -> str:
        cells = [pad(str(cell), i) for i, cell in enumerate(row)]
        text = indent + "  ".join(cells).rstrip()
        return dim(text, styled=styled) if dim_row else text

    lines: list[str] = []
    if header:
        lines.append(fmt(header, dim_row=True))
    for row in rows:
        lines.append(fmt(row))
    return lines
