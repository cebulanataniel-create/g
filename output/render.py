import sys, subprocess, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = sys.argv[1]
OUT = sys.argv[2]
W, H, FPS = 1920, 1080, 30
DUR = 66.0
XF = 0.8  # crossfade length
PREVIEW = len(sys.argv) > 3  # render single stills at given times

IMG = lambda n: f"{ROOT}/dl/img/{n}.jpg"
FONT_CAPS = f"{ROOT}/dl/fonts/f1.ttf"     # Cinzel
FONT_ITAL = f"{ROOT}/dl/fonts/f2.ttf"     # Cormorant Garamond Italic
FONT_REG = f"{ROOT}/dl/fonts/f3.ttf"      # Cormorant Garamond

# (image, start, end, (cx, cy, w) from, (cx, cy, w) to)  -- w = crop width as fraction of image width
SHOTS = [
    ("01_botticelli",    0.0,  5.2, (0.50, 0.50, 1.00), (0.49, 0.42, 0.80)),
    ("02_friedrich",     5.2,  9.6, (0.50, 0.82, 1.00), (0.50, 0.40, 1.00)),
    ("03_cot_spring",    9.6, 15.3, (0.50, 0.32, 0.92), (0.50, 0.23, 0.64)),
    ("04_leighton_june",15.3, 20.3, (0.50, 0.48, 1.00), (0.57, 0.33, 0.74)),
    ("01_botticelli",   20.3, 26.0, (0.48, 0.34, 0.52), (0.485, 0.30, 0.42)),
    ("06_raphael",      26.0, 31.5, (0.50, 0.52, 1.00), (0.50, 0.47, 0.55)),
    ("07_titian",       31.5, 37.0, (0.36, 0.45, 0.74), (0.64, 0.42, 0.74)),
    ("09_gerard",       37.0, 43.0, (0.50, 0.46, 1.00), (0.56, 0.26, 0.68)),
    ("10_michelangelo", 43.0, 49.8, (0.30, 0.50, 0.80), (0.56, 0.48, 0.55)),
    ("02_friedrich",    49.8, 55.2, (0.50, 0.40, 0.80), (0.50, 0.34, 0.62)),
    ("07_titian",       55.2, 60.3, (0.22, 0.46, 0.62), (0.18, 0.40, 0.52)),
    ("01_botticelli",   60.3, 66.0, (0.49, 0.44, 0.72), (0.50, 0.50, 1.00)),
]

# (start, end, italian, english)
LYRICS = [
    (9.7, 15.1, "O mio babbino caro,", "Oh, my dear papa,"),
    (15.4, 25.6, "mi piace, è bello, bello;", "I love him, he is so handsome"),
    (26.1, 36.7, "vo' andare in Porta Rossa", "I want to go to Porta Rossa"),
    (37.1, 49.5, "a comperar l'anello!", "to buy the ring!"),
    (50.0, 59.6, "Sì, sì, ci voglio andare!", "Yes, yes, I want to go there!"),
    (60.3, 64.6, "e se l'amassi indarno…", "and if my love were in vain…"),
]
FLASHES = [19.5, 48.5]  # high A-flat peaks

def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

# ---------- assets ----------
cache = {}
def load(name):
    if name not in cache:
        im = Image.open(IMG(name)).convert("RGB")
        cache[name] = im
    return cache[name]

def crop_box(im, cx, cy, w):
    iw, ih = im.size
    cw = w * iw
    ch = cw * H / W
    if ch > ih:  # too tall -> limit by height
        ch = ih; cw = ch * W / H
    x0 = min(max(cx * iw - cw / 2, 0), iw - cw)
    y0 = min(max(cy * ih - ch / 2, 0), ih - ch)
    return (x0, y0, x0 + cw, y0 + ch)

def shot_frame(shot, t):
    name, t0, t1, a, b = shot
    p = smooth((t - (t0 - XF / 2)) / ((t1 + XF / 2) - (t0 - XF / 2)))
    cx, cy, w = [a[i] + (b[i] - a[i]) * p for i in range(3)]
    # gentle "breathing" drift so it never feels static
    cx += 0.004 * math.sin(t * 0.9 + t0)
    cy += 0.004 * math.cos(t * 0.7 + t0)
    im = load(name)
    box = crop_box(im, cx, cy, w)
    fr = im.transform((W, H), Image.EXTENT, box, Image.BICUBIC)
    return np.asarray(fr, dtype=np.float32)

# vignette
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = (1 - 0.42 * np.clip(r - 0.45, 0, None) ** 1.6)[..., None]
# bottom gradient for lyric legibility
GRAD = (1 - 0.45 * np.clip((yy - H * 0.62) / (H * 0.38), 0, 1) ** 1.3)[..., None]
GRADE = np.array([1.04, 1.0, 0.92], dtype=np.float32)

rng = np.random.default_rng(7)
GRAIN = [rng.normal(0, 3.0, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1) for _ in range(6)]

# golden dust motes
def mote_sprite(size):
    s = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(s).ellipse((size, size, size * 3, size * 3), fill=255)
    return np.asarray(s.filter(ImageFilter.GaussianBlur(size * 0.6)), dtype=np.float32) / 255
random.seed(3)
MOTES = [dict(x=random.random() * W, y=random.random() * H, sp=random.uniform(12, 38),
              ph=random.random() * 6.28, sz=random.choice([3, 4, 5, 6]), a=random.uniform(0.25, 0.7))
         for _ in range(55)]
SPR = {s: mote_sprite(s) for s in (3, 4, 5, 6)}
GOLD = np.array([255, 222, 160], dtype=np.float32)

def add_motes(fr, t):
    for m in MOTES:
        x = (m["x"] + 18 * math.sin(t * 0.5 + m["ph"])) % W
        y = (m["y"] - m["sp"] * t) % H
        tw = 0.6 + 0.4 * math.sin(t * 2.1 + m["ph"] * 3)
        sp = SPR[m["sz"]]; sh = sp.shape[0]
        x0, y0 = int(x) - sh // 2, int(y) - sh // 2
        if x0 < 0 or y0 < 0 or x0 + sh > W or y0 + sh > H:
            continue
        al = (sp * m["a"] * tw)[..., None]
        reg = fr[y0:y0 + sh, x0:x0 + sh]
        fr[y0:y0 + sh, x0:x0 + sh] = reg + (GOLD - reg) * al * 0.8

# ---------- text ----------
def text_layer(lines):
    """lines: list of (text, fontpath, size, y_center, fill, tracking)"""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    for text, fp, size, yc, fill, track in lines:
        f = ImageFont.truetype(fp, size)
        if track:
            widths = [f.getlength(ch) for ch in text]
            total = sum(widths) + track * (len(text) - 1)
            x = (W - total) / 2
            for ch, cw in zip(text, widths):
                ds.text((x + 2, yc + 3), ch, font=f, fill=(0, 0, 0, 255), anchor="lm")
                d.text((x, yc), ch, font=f, fill=fill, anchor="lm")
                x += cw + track
        else:
            ds.text((W / 2 + 2, yc + 3), text, font=f, fill=(0, 0, 0, 255), anchor="mm")
            d.text((W / 2, yc), text, font=f, fill=fill, anchor="mm")
    shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    out = Image.alpha_composite(shadow, layer)
    a = np.asarray(out, dtype=np.float32)
    return a[..., :3], a[..., 3:] / 255

CREAM = (247, 238, 220, 255)
GOLDT = (226, 196, 140, 255)
TEXTS = []
title = text_layer([
    ("O MIO BABBINO CARO", FONT_CAPS, 92, H * 0.44, CREAM, 10),
    ("— Giacomo Puccini · Gianni Schicchi, 1918 —", FONT_ITAL, 50, H * 0.56, (240, 214, 160, 255), 0),
])
TEXTS.append((0.6, 4.7, title, 0.8))
for s, e, it, en in LYRICS:
    TEXTS.append((s, e, text_layer([
        (it, FONT_ITAL, 70, H * 0.835, CREAM, 0),
        (en.upper(), FONT_REG, 33, H * 0.907, (240, 214, 160, 255), 4),
    ]), 0.45))
credit = text_layer([
    ("Lupita Pérez Arias, soprano · CC BY 3.0", FONT_ITAL, 32, H * 0.92, (235, 225, 205, 255), 0),
])
TEXTS.append((64.8, 66.0, credit, 0.4))

def comp_text(fr, t):
    for s, e, (rgb, al), fd in TEXTS:
        if s - 0.01 <= t <= e:
            k = min(smooth((t - s) / fd), smooth((e - t) / fd))
            # subtle rise-in
            if k <= 0:
                continue
            fr[:] = fr + (rgb - fr) * (al * k)

# ---------- frame ----------
def frame(t):
    active = [s for s in SHOTS if s[1] - XF / 2 <= t <= s[2] + XF / 2]
    fr = None
    for s in active:
        img = shot_frame(s, t)
        if fr is None:
            fr = img
        else:
            k = smooth((t - (s[1] - XF / 2)) / XF)
            fr = fr * (1 - k) + img * k
    fr = fr * GRADE
    # luminous flash on climactic notes
    for ft in FLASHES:
        d = t - ft
        if -0.15 < d < 1.2:
            g = (d + 0.15) / 0.15 if d < 0 else math.exp(-d * 3.2)
            fr = fr + (255 - fr) * 0.33 * g * np.array([1, 0.93, 0.8], dtype=np.float32)
    fr = fr * VIG * GRAD
    add_motes(fr, t)
    comp_text(fr, t)
    fr = fr + GRAIN[int(t * FPS) % len(GRAIN)]
    # fade from/to black
    fade = min(smooth(t / 1.0), smooth((DUR - t) / 1.6))
    fr = fr * fade
    return np.clip(fr, 0, 255).astype(np.uint8)

if PREVIEW:
    for ts in sys.argv[3].split(","):
        Image.fromarray(frame(float(ts))).save(f"{OUT}_{ts}.jpg", quality=85)
    sys.exit()

ff = subprocess.Popen([
    "ffmpeg", "-v", "error", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
    "-ss", "0", "-t", str(DUR), "-i", f"{ROOT}/dl/audio/lupita.mp3",
    "-filter:a", f"afade=t=in:d=0.4,afade=t=out:st={DUR-2.6}:d=2.6",
    "-c:v", "libx264", "-preset", "medium", "-b:v", "5M", "-maxrate", "7M", "-bufsize", "10M", "-tune", "film", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", OUT,
], stdin=subprocess.PIPE)
N = int(DUR * FPS)
for i in range(N):
    ff.stdin.write(frame(i / FPS).tobytes())
    if i % 150 == 0:
        print(f"{i}/{N}", flush=True)
ff.stdin.close()
ff.wait()
print("done", ff.returncode)
