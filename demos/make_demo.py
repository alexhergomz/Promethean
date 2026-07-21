#!/usr/bin/env python3
"""
Generate animated GIF demo of promethean using PIL.
Simulates a realistic terminal session with tool calls.
"""
from PIL import Image, ImageDraw, ImageFont
import os, textwrap

# ── Catppuccin Mocha palette ─────────────────────────────────────────────
BG      = (30,  30,  46)   # base
SURFACE = (49,  50,  68)   # surface0
TEXT    = (205, 214, 244)  # text
SUBTEXT = (108, 112, 134)  # overlay0 (dim)
CYAN    = (137, 220, 235)  # sky
GREEN   = (166, 227, 161)  # green
YELLOW  = (249, 226, 175)  # yellow
RED     = (243, 139, 168)  # red
MAUVE   = (203, 166, 247)  # mauve (user prompt)
BLUE    = (137, 180, 250)  # blue
PEACH   = (250, 179, 135)  # peach

W, H = 960, 720
FONT_PATH  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SIZE  = 14
LINE_H     = 20
PAD_X      = 18
PAD_Y      = 16


def make_font(size=FONT_SIZE, bold=False):
    path = FONT_BOLD if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()


FONT      = make_font()
FONT_B    = make_font(bold=True)
FONT_SM   = make_font(FONT_SIZE - 1)


# ── Segment: (text, color, bold?) ────────────────────────────────────────
Seg = tuple   # (str, rgb_tuple, bool)


def seg(t, c=TEXT, b=False): return (t, c, b)
def segs(*args): return list(args)


def render_line(draw, y, segments, x_start=PAD_X):
    x = x_start
    for text, color, bold in segments:
        font = FONT_B if bold else FONT
        draw.text((x, y), text, font=font, fill=color)
        x += font.getlength(text)
    return y + LINE_H


def blank_frame():
    img = Image.new("RGB", (W, H), BG)
    return img


def draw_frame(lines_segments):
    """
    lines_segments: list of either
      - list[Seg]  → rendered as a line
      - None       → blank line
    Returns PIL Image.
    """
    img = blank_frame()
    d   = ImageDraw.Draw(img)
    y = PAD_Y
    for item in lines_segments:
        if item is None:
            y += LINE_H
        elif isinstance(item, list):
            y = render_line(d, y, item)
        else:
            y = render_line(d, y, [item])
    return img


# ── Pre-defined screen content blocks ───────────────────────────────────

BANNER = [
    [seg("╭─ Promethean ────────────────────────────────────────────────╮", SUBTEXT)],
    [seg("│  ", SUBTEXT), seg("Model: ", SUBTEXT), seg("custom/qwen3.5-9b", PEACH, True),
     seg("    Backend: ", SUBTEXT), seg("llama.cpp", CYAN)],
    [seg("│  ", SUBTEXT), seg("Context: ", SUBTEXT), seg("57K", CYAN),
     seg("    Permissions: ", SUBTEXT), seg("auto", YELLOW),
     seg("    local — no keys", SUBTEXT)],
    [seg("│  Type /help for commands · ESC to interrupt · Ctrl+C to quit  │", SUBTEXT)],
    [seg("╰──────────────────────────────────────────────────────────────╯", SUBTEXT)],
    None,
]

def prompt_line(text="", cursor=False):
    cur = "█" if cursor else ""
    return [
        seg("[promethean] ", SUBTEXT),
        seg("» ", CYAN, True),
        seg(text + cur, TEXT),
    ]

def asst_header():
    return [
        seg("╭─ qwen3.5-9b ", SUBTEXT),
        seg("●", GREEN),
        seg(" ────────────────────────────────────────────", SUBTEXT),
    ]

def asst_sep():
    return [seg("╰──────────────────────────────────────────────────────────", SUBTEXT)]

def footer(pct=8, tps=34):
    return [
        seg(" custom/qwen3.5-9b", PEACH),
        seg(" · ", SUBTEXT), seg("ctx ", SUBTEXT), seg(f"{pct}%", GREEN),
        seg(" · ", SUBTEXT), seg("auto", YELLOW),
        seg(" · ", SUBTEXT), seg(f"{tps} t/s", GREEN),
    ]

def tool_line(icon, name, arg, color=CYAN):
    return [
        seg(f"  {icon}  ", SUBTEXT),
        seg(name, color),
        seg("(", SUBTEXT),
        seg(arg, TEXT),
        seg(")", SUBTEXT),
    ]

def tool_ok(msg):
    return [seg("  ✓ ", GREEN), seg(msg, SUBTEXT)]

def tool_err(msg):
    return [seg("  ✗ ", RED), seg(msg, SUBTEXT)]

def diff_del(t):
    return [seg("    - ", RED), seg(t, RED)]

def diff_add(t):
    return [seg("    + ", GREEN), seg(t, GREEN)]

def text_line(t, indent=2):
    return [seg(" " * indent + t, TEXT)]

def dim_line(t, indent=4):
    return [seg(" " * indent + t, SUBTEXT)]


# ── Scene builder ─────────────────────────────────────────────────────────
#
# The demo runs entirely against the local llama.cpp backend and showcases the
# two things that make a weak local model usable: Edit recovers from a
# non-verbatim match, and the edited file is auto-verified so the model fixes
# its own mistake on the same turn. It closes on /model recommend.

def build_scenes():
    """Return list of (frame_content, duration_ms)."""
    scenes = []
    def add(lines, ms=120):
        scenes.append((lines, ms + 0))   # ms passed through; kept explicit

    # ── Scene 0: banner ──────────────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True), None, footer(6, 0)], 800)

    # ── Scene 1: user types the request ──────────────────────────────────
    msg1 = "Fix the off-by-one in paginate() in utils.py"
    for i in range(0, len(msg1) + 1, 3):
        add(BANNER + [prompt_line(msg1[:i], cursor=(i < len(msg1))), None, footer(6, 0)], 55)
    add(BANNER + [prompt_line(msg1), None, footer(6, 0)], 350)

    pre  = BANNER + [prompt_line(msg1)]
    base = pre + [None, asst_header()]

    # ── Scene 2: Edit call, recovered from a non-verbatim match ──────────
    edit1 = [
        tool_line("✎", "Edit", "utils.py", MAUVE),
    ]
    add(base + edit1, 500)
    edit1_done = [
        tool_line("✎", "Edit", "utils.py", MAUVE),
        diff_del("return items[start : start + size + 1]"),
        diff_add("return items[start : start + size]"),
        dim_line("(no verbatim match — applied via indentation-insensitive match)", 2),
    ]
    add(base + edit1_done, 700)

    # ── Scene 3: auto verify-after-edit catches a second bug ─────────────
    verify_bad = edit1_done + [
        None,
        [seg("  [verify] ", PEACH), seg("pyright — 1 issue:", SUBTEXT)],
        [seg("    12:16 ", SUBTEXT), seg("[error] ", RED), seg("\"size\" is possibly unbound", TEXT)],
    ]
    add(base + verify_bad, 900)

    # ── Scene 4: model fixes the unbound default, verify clean ───────────
    fix = verify_bad + [
        None,
        tool_line("✎", "Edit", "utils.py", MAUVE),
        diff_add("size = size or 20"),
        [seg("  [verify] ", PEACH), seg("pyright: no issues", GREEN)],
    ]
    add(base + fix, 500)
    add(base + fix, 700)

    # ── Scene 5: assistant wraps up ──────────────────────────────────────
    resp = [
        "Fixed two things in paginate():",
        "",
        "  • the slice was one past the page (off-by-one)",
        "  • size had no default, so an unset size raised — added size = size or 20",
        "",
        "The verifier confirms utils.py is clean.",
    ]
    section = fix + [None, [seg("│ ", SUBTEXT)]]
    streamed = []
    for rline in resp:
        streamed.append(text_line(rline, 2))
        add(base + section + streamed + [None, footer(11, 41)], 80 if rline else 30)
    full1 = base + section + [text_line(l, 2) for l in resp] + [asst_sep(), None]
    add(full1 + [prompt_line(cursor=True), None, footer(11, 41)], 900)

    # ── Scene 6: /model recommend ────────────────────────────────────────
    slash = "/model recommend"
    for i in range(0, len(slash) + 1, 2):
        add(full1 + [prompt_line(slash[:i], cursor=(i < len(slash))), None, footer(11, 41)], 55)
    add(full1 + [prompt_line(slash), None, footer(11, 41)], 350)

    rec = [
        [seg("Hardware:  ", SUBTEXT), seg("16 GB RAM", TEXT), seg(" · ", SUBTEXT), seg("8.6 GB VRAM", TEXT)],
        [seg("Budget:    ", SUBTEXT), seg("8.6 GB", CYAN), seg("  (GPU-resident)", SUBTEXT)],
        None,
        [seg("Recommended for ~9 GB", TEXT, True),
         seg("  (quant sized to fit the full context)", SUBTEXT)],
        None,
        [seg("  qwen3.5-9b", CYAN), seg(" ★  ", YELLOW), seg("Qwen3.5", SUBTEXT),
         seg("  max 57K   ", SUBTEXT), seg("[flagship]", SUBTEXT)],
        [seg("      → ", SUBTEXT), seg("UD-Q4_K_XL (5.97 GB) · fits ~79K ctx", GREEN)],
        [seg("        huggingface.co/unsloth/Qwen3.5-9B-GGUF", SUBTEXT)],
        [seg("  qwen3.5-4b", CYAN), seg("     Qwen3.5", SUBTEXT), seg("  max 57K", SUBTEXT)],
        [seg("      → ", SUBTEXT), seg("UD-Q8_K_XL (5.95 GB) · fits ~160K ctx", GREEN)],
    ]
    grow = full1 + [prompt_line(slash), None]
    acc = []
    for rline in rec:
        acc.append(rline if rline is not None else None)
        add(grow + acc + [None, footer(11, 41)], 110)
    add(grow + rec + [None, prompt_line(cursor=True), None, footer(11, 41)], 2200)

    return scenes


# ── Render ────────────────────────────────────────────────────────────────

def _build_explicit_palette():
    """
    Build a 256-entry palette from our exact theme colors.
    Returns flat list of 768 ints (R,G,B, R,G,B, ...) suitable for putpalette().
    """
    # All distinct colors used in the renderer
    theme = [
        BG, SURFACE, TEXT, SUBTEXT,
        CYAN, GREEN, YELLOW, RED, MAUVE, BLUE, PEACH,
        (255, 255, 255), (0, 0, 0),
        # Extra intermediate shades that PIL might snap to
        (50, 55, 80),   # surface variant
        (90, 95, 120),  # dim text variant
        (160, 166, 200),
    ]
    flat = []
    for c in theme:
        flat.extend(c)
    # Pad to 256 entries with black
    while len(flat) < 256 * 3:
        flat.extend((0, 0, 0))
    return flat


def render_gif(output_path="demo.gif"):
    print("Building scenes...")
    scenes = build_scenes()
    print(f"  {len(scenes)} scenes")

    palette_data = _build_explicit_palette()

    # Create a palette-mode reference image for quantize()
    pal_ref = Image.new("P", (1, 1))
    pal_ref.putpalette(palette_data)

    print("  Rendering frames...")
    rgb_frames = []
    durations  = []
    for i, (lines, ms) in enumerate(scenes):
        img = draw_frame(lines)
        rgb_frames.append(img)
        durations.append(ms)
        if i % 20 == 0:
            print(f"  {i}/{len(scenes)}...")

    # Quantize all frames to the same explicit palette (no dither → exact snap)
    print("  Quantizing to global palette...")
    p_frames = [f.quantize(palette=pal_ref, dither=0) for f in rgb_frames]

    print(f"Saving GIF → {output_path}  ({len(p_frames)} frames)...")
    p_frames[0].save(
        output_path,
        save_all=True,
        append_images=p_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    size_kb = os.path.getsize(output_path) // 1024
    print(f"Done! {size_kb} KB")


# ── Static screenshot ─────────────────────────────────────────────────────

def render_screenshot(output_path="screenshot.png"):
    """Single high-quality screenshot showing a complete session."""
    lines = (
        BANNER +
        [prompt_line("Fix the off-by-one in paginate() in utils.py")] +
        [None, asst_header()] +
        [
            tool_line("✎", "Edit", "utils.py", MAUVE),
            diff_del("return items[start : start + size + 1]"),
            diff_add("return items[start : start + size]"),
            dim_line("(no verbatim match — applied via indentation-insensitive match)", 2),
            None,
            [seg("  [verify] ", PEACH), seg("pyright — 1 issue:", SUBTEXT)],
            [seg("    12:16 ", SUBTEXT), seg("[error] ", RED), seg("\"size\" is possibly unbound", TEXT)],
            None,
            tool_line("✎", "Edit", "utils.py", MAUVE),
            diff_add("size = size or 20"),
            [seg("  [verify] ", PEACH), seg("pyright: no issues", GREEN)],
            None,
            [seg("│ ", SUBTEXT)],
            text_line("Fixed the off-by-one and the unbound `size`. The verifier", 2),
            text_line("confirms utils.py is clean.", 2),
            asst_sep(),
            None,
            prompt_line(cursor=True),
            None,
            footer(11, 41),
        ]
    )
    img = draw_frame(lines)

    # Add subtle rounded border effect
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W-1, H-1], outline=SURFACE, width=2)

    img.save(output_path, format="PNG", optimize=True)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"Screenshot saved: {output_path}  ({size_kb} KB)")


if __name__ == "__main__":
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")

    gif_path = os.path.join(docs_dir, "demo.gif")
    png_path = os.path.join(docs_dir, "screenshot.png")

    render_screenshot(png_path)
    render_gif(gif_path)
    print("\nFiles created:")
    print(f"  {png_path}")
    print(f"  {gif_path}")
