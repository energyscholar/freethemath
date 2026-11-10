#!/usr/bin/env python3
"""Generate every figure on freethemath.org from simulated or schematic data.

Run from the repo root:  python3 figures-src/make_figures.py

Writes SVG figures to img/ and the social card to og.png (repo root).
Deterministic: fixed seed 1465, SVG hash salt fixed, no date metadata.
Needs numpy, scipy and matplotlib.
"""

import logging
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Glyph .* missing from font")

SEED = 1465
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "img")

# ---- palette: the house tutorial slate theme --------------------------------
BG = "#1e293b"       # card
INK = "#cbd5e1"      # body text
HEAD = "#f1f5f9"     # headings
MUTED = "#94a3b8"    # axis labels, ticks
GRID = "#334155"     # grid lines, spines
BLUE = "#3b82f6"
RED = "#ef4444"
AMBER = "#f59e0b"
PURPLE = "#a855f7"
CYAN = "#06b6d4"
GREEN = "#22c55e"
PINK = "#ec4899"

FIELD = {
    "physics": BLUE,
    "heart": RED,
    "finance": AMBER,
    "ml": PURPLE,
    "grid": CYAN,
    "traffic": GREEN,
}

matplotlib.rcParams.update({
    "svg.hashsalt": "1465",
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["system-ui", "Segoe UI", "Roboto", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": HEAD,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "path.simplify": True,
})


def rng_for(k):
    """One independent, reproducible stream per figure."""
    return np.random.default_rng([SEED, k])


def save_svg(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    return path


# ---- F0: the hero -----------------------------------------------------------

def fold_series(rng):
    """Noisy saddle-node normal form dx/dt = mu - x^2, with mu falling to 0.

    The stable state sits at sqrt(mu); its recovery rate 2*sqrt(mu) falls
    toward 0 as mu -> 0, then the state vanishes and the system tips.
    """
    dt = 0.01
    n = 16000
    mu = np.linspace(1.0, -0.2, n)
    sigma = 0.07
    x = np.empty(n)
    x[0] = 1.0
    noise = rng.standard_normal(n) * sigma * np.sqrt(dt)
    for t in range(1, n):
        xt = x[t - 1]
        x[t] = xt + (mu[t - 1] - xt * xt) * dt + noise[t]
        if x[t] < -3.0:
            x[t:] = np.nan
            break
    progress = np.linspace(0.0, 1.0, n)      # control parameter, rising
    return progress, mu, x


def rolling_indicators(x, valid, window):
    """Rolling variance and lag-1 autocorrelation of the detrended series."""
    xs = x[:valid]
    trend = gaussian_filter1d(xs, sigma=300, mode="nearest")
    resid = xs - trend
    w = np.lib.stride_tricks.sliding_window_view(resid, window)
    var = w.var(axis=1)
    a = w[:, :-1] - w[:, :-1].mean(axis=1, keepdims=True)
    b = w[:, 1:] - w[:, 1:].mean(axis=1, keepdims=True)
    ac1 = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    idx = np.arange(window - 1, valid)
    return idx, var, ac1


def draw_valleys(ax, labels=True):
    xs = np.linspace(-1.35, 1.6, 300)
    mus = [1.0, 0.45, 0.12, 0.0]
    shades = [BLUE, "#6d8fe0", AMBER, RED]
    names = ["far from the tip", "closer", "almost there", "tipping point"]
    for i, (m, c) in enumerate(zip(mus, shades)):
        off = -1.6 * i
        v = xs ** 3 / 3.0 - m * xs
        v = v - (np.sqrt(m) ** 3 / 3.0 - m * np.sqrt(m))   # well bottom at 0
        ax.plot(xs, v + off, color=c, lw=2.2, solid_capstyle="round")
        xb = np.sqrt(m)
        ax.plot([xb], [off + 0.13], "o", ms=9, color=HEAD, mec=c, mew=2)
        if labels:
            ax.text(-1.32, off + 0.55, names[i], color=c, fontsize=9.5,
                    ha="left", va="bottom")
    ax.set_xlim(-1.4, 1.65)
    ax.set_ylim(-5.9, 1.4)
    ax.axis("off")


def draw_hero(fig, axes, labels=True):
    ax_v, ax_x, ax_var, ax_ac = axes
    rng = rng_for(0)
    p, mu, x = fold_series(rng)
    tip = int(np.argmin(np.abs(mu)))                 # where mu crosses 0
    valid = int(np.argmax(np.isnan(x))) if np.isnan(x).any() else len(x)
    stop = min(tip, valid) - 300
    idx, var, ac1 = rolling_indicators(x, stop, window=2000)

    draw_valleys(ax_v, labels=labels)

    step = 10
    xx = x.copy()
    ax_x.plot(p[::step], np.clip(xx, -1.6, None)[::step], color=INK, lw=0.9)
    ax_var.plot(p[idx][::step], var[::step], color=AMBER, lw=2)
    ax_ac.plot(p[idx][::step], ac1[::step], color=RED, lw=2)
    for a in (ax_x, ax_var, ax_ac):
        a.axvline(p[tip], color=RED, lw=1.2, ls=(0, (4, 3)), alpha=0.9)
        a.set_xlim(0, min(1.0, p[valid - 1] + 0.1))
        a.tick_params(labelbottom=False)
        a.set_yticks([])
    ax_x.set_ylim(-1.7, 1.5)
    if labels:
        ax_x.set_ylabel("state", color=INK)
        ax_var.set_ylabel("variance", color=AMBER)
        ax_ac.set_ylabel("memory", color=RED)
        ax_ac.set_xlabel("pressure on the system rises  →", color=INK)
        ax_x.text(p[tip] - 0.01, 1.25, "tipping point", color=RED,
                  fontsize=9.5, ha="right", va="center")
        ax_x.text(p[valid - 1] + 0.012, -1.3, "tipped", color=MUTED,
                  fontsize=9, ha="left", va="center")
        ax_v.set_title("The valley flattens", loc="left", pad=6)
        ax_x.set_title("The warning signs rise first", loc="left", pad=6)


def figure_hero():
    # wide layout (desktop)
    fig = plt.figure(figsize=(9.6, 4.8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.55], hspace=0.18,
                          wspace=0.12, left=0.02, right=0.98, top=0.9,
                          bottom=0.1)
    ax_v = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_var = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_ac = fig.add_subplot(gs[2, 1], sharex=ax_x)
    draw_hero(fig, (ax_v, ax_x, ax_var, ax_ac))
    out = [save_svg(fig, "f0-hero.svg")]

    # tall layout (phones)
    fig = plt.figure(figsize=(4.6, 7.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.9, 1, 0.8, 0.8], hspace=0.22,
                          left=0.1, right=0.97, top=0.95, bottom=0.06)
    ax_v = fig.add_subplot(gs[0])
    ax_x = fig.add_subplot(gs[1])
    ax_var = fig.add_subplot(gs[2], sharex=ax_x)
    ax_ac = fig.add_subplot(gs[3], sharex=ax_x)
    draw_hero(fig, (ax_v, ax_x, ax_var, ax_ac))
    out.append(save_svg(fig, "f0-hero-tall.svg"))

    # social card: 1200x630, no text
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.55], hspace=0.12,
                          wspace=0.08, left=0.04, right=0.97, top=0.93,
                          bottom=0.07)
    ax_v = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_var = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_ac = fig.add_subplot(gs[2, 1], sharex=ax_x)
    draw_hero(fig, (ax_v, ax_x, ax_var, ax_ac), labels=False)
    for a in (ax_x, ax_var, ax_ac):
        a.grid(False)
    og = os.path.join(ROOT, "og.png")
    fig.savefig(og, format="png", dpi=100,
                metadata={"Software": None})
    plt.close(fig)
    out.append(og)
    return out


# ---- F1: physics -- correlation length --------------------------------------

def ising_snapshot(rng, temp, size=64, sweeps=1500, ordered=False):
    """Checkerboard Metropolis on a periodic 2D Ising lattice."""
    s = np.ones((size, size), dtype=np.int8) if ordered else \
        rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))
    ii, jj = np.indices((size, size))
    masks = [((ii + jj) % 2 == c) for c in (0, 1)]
    beta = 1.0 / temp
    for _ in range(sweeps):
        for m in masks:
            nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) +
                  np.roll(s, 1, 1) + np.roll(s, -1, 1)).astype(np.int16)
            de = 2 * s.astype(np.int16) * nb
            flip = (de <= 0) | (rng.random(s.shape) < np.exp(-beta * de))
            s = np.where(m & flip, -s, s).astype(np.int8)
    return s


def figure_physics():
    rng = rng_for(1)
    tc = 2.0 / np.log(1 + np.sqrt(2))
    temps = [(1.9, True), (tc, False), (2.9, False)]
    snaps = [ising_snapshot(rng, t, ordered=o) for t, o in temps]

    fig = plt.figure(figsize=(4.6, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.42,
                          wspace=0.08, left=0.1, right=0.97, top=0.92,
                          bottom=0.04)
    ax = fig.add_subplot(gs[0, :])
    t = np.linspace(1.5, 3.1, 600)
    # schematic: exponent nu = 1, rounded off at the peak as in any finite sample
    xi = 1.0 / np.sqrt((t / tc - 1.0) ** 2 + 0.026 ** 2)
    ax.plot(t, xi, color=BLUE, lw=2.2)
    ax.axvline(tc, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(tc + 0.04, 34, "critical point", color=RED, fontsize=9)
    ax.set_xlim(1.5, 3.1)
    ax.set_ylim(0, 42)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("temperature →", labelpad=2)
    ax.set_ylabel("correlation length")
    ax.set_title("Physics: correlation length", loc="left")
    names = ["below", "at", "above"]
    for k, (snap, nm) in enumerate(zip(snaps, names)):
        a = fig.add_subplot(gs[1, k])
        a.imshow(snap, cmap=matplotlib.colors.ListedColormap([BG, BLUE]),
                 interpolation="none", vmin=-1, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_color(RED if nm == "at" else GRID)
        a.set_title(nm, fontsize=9.5, color=RED if nm == "at" else MUTED,
                    fontweight="normal", pad=3)
    return [save_svg(fig, "f1-physics.svg")]


# ---- long-memory noise, DFA and R/S (F2, F3) --------------------------------

def fgn(rng, n, hurst):
    """Fractional Gaussian noise by exact spectral (Davies-Harte) synthesis."""
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * k ** (2 * hurst) +
               (k + 1) ** (2 * hurst))
    row = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(row).real
    lam = np.clip(lam, 0, None)
    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    y = np.fft.fft(np.sqrt(lam / m) * z)
    return y.real[:n]


def dfa(x, scales):
    y = np.cumsum(x - x.mean())
    out = []
    for s in scales:
        nseg = len(y) // s
        seg = y[: nseg * s].reshape(nseg, s)
        t = np.arange(s)
        coef = np.polyfit(t, seg.T, 1)
        fit = np.outer(coef[0], t) + coef[1][:, None]
        out.append(np.sqrt(((seg - fit) ** 2).mean()))
    return np.array(out)


def figure_heart():
    rng = rng_for(2)
    n = 2 ** 14
    scales = np.unique(np.logspace(np.log10(8), np.log10(1024), 16).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.15)
    cases = [(0.5, MUTED, "uncorrelated"), (0.75, AMBER, ""),
             (1.0, RED, "long-range (critical)")]
    for h, c, nm in cases:
        x = fgn(rng, n, min(h, 0.99))
        f = dfa(x, scales)
        slope = np.polyfit(np.log(scales), np.log(f), 1)[0]
        lab = f"α ≈ {slope:.2f}" + (f"  {nm}" if nm else "")
        ax.loglog(scales, f / f[0], "o-", color=c, ms=3.5, lw=1.8, label=lab)
    ax.set_xlabel("window size n (beats)")
    ax.set_ylabel("fluctuation F(n)")
    ax.set_xticks([10, 100, 1000])
    ax.set_xticklabels(["10", "100", "1000"])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.legend(loc="upper left", labelcolor=INK)
    ax.set_title("Heart rate: DFA scaling exponent α", loc="left")
    return [save_svg(fig, "f2-heart.svg")]


def rescaled_range(x, scales):
    out = []
    for s in scales:
        nseg = len(x) // s
        seg = x[: nseg * s].reshape(nseg, s)
        dev = np.cumsum(seg - seg.mean(axis=1, keepdims=True), axis=1)
        r = dev.max(axis=1) - dev.min(axis=1)
        sd = seg.std(axis=1)
        out.append((r / sd).mean())
    return np.array(out)


def figure_finance():
    rng = rng_for(3)
    n = 2 ** 13
    scales = np.unique(np.logspace(np.log10(16), np.log10(1024), 10).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.15)
    cases = [(0.5, MUTED), (0.75, "#d9a441"), (0.95, AMBER)]
    show = 1500
    for i, (h, c) in enumerate(cases):
        x = fgn(rng, n, h)
        rs = rescaled_range(x, scales)
        hest = np.polyfit(np.log(scales), np.log(rs), 1)[0]
        path = np.cumsum(x[:show])
        path = (path - path.mean()) / (path.std() + 1e-12)
        off = -3.6 * i
        ax.plot(np.arange(show)[::3], path[::3] + off, color=c, lw=1.3)
        ax.text(show * 1.02, off, f"H ≈ {hest:.2f}", color=c,
                fontsize=9.5, va="center")
    ax.set_xlim(0, show * 1.23)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time →   (memory grows from top to bottom)")
    ax.set_title("Finance: Hurst exponent H (market memory)", loc="left")
    return [save_svg(fig, "f3-finance.svg")]


# ---- F4: machine learning -- spectral radius --------------------------------

def figure_ml():
    rng = rng_for(4)
    n = 200
    w0 = rng.standard_normal((n, n)) / np.sqrt(n)
    rad0 = np.max(np.abs(np.linalg.eigvals(w0)))
    w0 = w0 / rad0
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.88, bottom=0.15)
    steps = 120
    pulse = rng.standard_normal(n) * 1e-3
    cases = [(0.5, MUTED), (0.8, "#c084fc"), (0.95, PURPLE), (0.99, PINK)]
    for chi, c in cases:
        w = w0 * chi
        x = pulse.copy()
        amp = [np.linalg.norm(x)]
        for _ in range(steps):
            x = np.tanh(w @ x)
            amp.append(np.linalg.norm(x))
        amp = np.array(amp) / amp[0]
        ax.semilogy(np.arange(steps + 1), amp, color=c, lw=2)
        lab = f"χ = {chi:g}"
        below = np.nonzero(amp < 1e-5)[0]
        if below.size:
            ax.text(below[0] + 3, 1e-5, lab, color=c, fontsize=9.5,
                    va="center", ha="left")
        else:
            ax.text(steps - 1, amp[-1] * 3.0, lab, color=c, fontsize=9.5,
                    va="bottom", ha="right")
    ax.set_ylim(1e-6, 3)
    ax.set_xlim(0, steps)
    ax.set_xlabel("time steps after a single input pulse")
    ax.set_ylabel("echo left")
    ax.set_yticks([1, 1e-3, 1e-6])
    ax.set_yticklabels(["full", "1/1000", "1/10⁶"])
    ax.minorticks_off()
    ax.set_title("Machine learning: spectral radius χ → 1", loc="left")
    return [save_svg(fig, "f4-ml.svg")]


# ---- F5: power grids -- cascade sizes ---------------------------------------

def cascade_sizes(rng, lam, runs, cap):
    """Total size of a Poisson branching cascade: each failure trips on
    average `lam` more. lam -> 1 is the critical load."""
    sizes = np.empty(runs, dtype=np.int64)
    for r in range(runs):
        total = 1
        active = 1
        while active and total < cap:
            active = rng.poisson(lam * active)
            total += active
        sizes[r] = min(total, cap)
    return sizes


def figure_grid():
    rng = rng_for(5)
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.15)
    cases = [(0.5, MUTED, "light load"), (0.8, "#67c6d8", ""),
             (0.95, CYAN, ""), (1.0, RED, "critical load")]
    cap = 10 ** 5
    for lam, c, nm in cases:
        s = cascade_sizes(rng, lam, 20000, cap)
        vals = np.unique(s)
        ccdf = np.array([(s >= v).mean() for v in vals])
        keep = vals < cap
        lab = f"load {lam:g}" + (f"  ({nm})" if nm else "")
        ax.loglog(vals[keep], ccdf[keep], color=c, lw=2, label=lab)
    ax.set_xlim(1, 3e4)
    ax.set_ylim(3e-5, 1.3)
    ax.set_xlabel("blackout size s (lines lost)")
    ax.set_ylabel("chance of a blackout ≥ s")
    ax.minorticks_off()
    ax.legend(loc="lower left", labelcolor=INK)
    ax.set_title("Power grids: cascade sizes", loc="left")
    return [save_svg(fig, "f5-grid.svg")]


# ---- F6: traffic -- flow-density diagram ------------------------------------

def nasch_flow(rng, density, length=1000, vmax=5, p_slow=0.25,
               warm=300, steps=300):
    """Nagel-Schreckenberg cellular automaton on a ring road."""
    ncar = max(1, int(round(density * length)))
    pos = np.sort(rng.choice(length, ncar, replace=False))
    vel = np.zeros(ncar, dtype=np.int64)
    moved = 0
    for t in range(warm + steps):
        gap = (np.roll(pos, -1) - pos - 1) % length
        if ncar == 1:
            gap = np.array([length - 1])
        vel = np.minimum(vel + 1, vmax)
        vel = np.minimum(vel, gap)
        slow = rng.random(ncar) < p_slow
        vel = np.where(slow, np.maximum(vel - 1, 0), vel)
        pos = (pos + vel) % length
        order = np.argsort(pos, kind="stable")
        pos, vel = pos[order], vel[order]
        if t >= warm:
            moved += vel.sum()
    return moved / (length * steps)


def figure_traffic():
    rng = rng_for(6)
    dens = np.concatenate([np.linspace(0.01, 0.2, 20),
                           np.linspace(0.22, 0.9, 18)])
    flow = np.array([nasch_flow(rng, d) for d in dens])
    peak = dens[np.argmax(flow)]
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.15)
    free = dens <= peak
    ax.plot(dens[free], flow[free], "o-", color=GREEN, ms=4, lw=1.6)
    ax.plot(dens[~free], flow[~free], "o-", color=RED, ms=4, lw=1.6)
    ax.axvline(peak, color=AMBER, lw=1.2, ls=(0, (4, 3)))
    ax.text(peak + 0.015, flow.max() * 1.02, "critical density",
            color=AMBER, fontsize=9, va="bottom")
    mid = len(dens[free]) // 2
    ax.annotate("free flow", xy=(dens[free][mid], flow[free][mid]),
                xytext=(0.2, flow.max() * 0.25), color=GREEN, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1))
    ax.text(0.55, flow.max() * 0.62, "jams", color=RED, fontsize=10)
    ax.set_ylim(0, flow.max() * 1.18)
    ax.set_xlim(0, 0.92)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("density (cars per km) →")
    ax.set_ylabel("flow (cars per hour)")
    ax.set_title("Traffic: the flow–density diagram", loc="left")
    return [save_svg(fig, "f6-traffic.svg")]


# ---- F7: the convergence timeline -------------------------------------------

ROWS = [
    ("Traffic flow", FIELD["traffic"], [1935, 2004]),
    ("Physics", FIELD["physics"], [1944, 1971]),
    ("Heart rate", FIELD["heart"], [1994]),
    ("Finance", FIELD["finance"], [1994]),
    ("Machine learning", FIELD["ml"], [2001]),
    ("Power grids", FIELD["grid"], [2004]),
]


def draw_timeline(fig, ax, compact=False):
    ax.axvspan(2010, 2025, color=GRID, alpha=0.45, lw=0)
    ax.text(2017.5, len(ROWS) - 0.35,
            "contact\nbegins" if compact else "cross-field\ncontact begins",
            ha="center", va="top", color=MUTED,
            fontsize=8.5 if compact else 9)
    for i, (name, c, years) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        ax.plot([years[0], 2025], [y, y], color=c, lw=3 if compact else 3.5,
                alpha=0.35, solid_capstyle="butt")
        for yr in years:
            ax.plot([yr], [y], "o", ms=8, color=c, mec=HEAD, mew=1.2)
            ax.text(yr, y + 0.3, str(yr), ha="center", va="bottom",
                    color=c, fontsize=8.5 if compact else 9)
        ax.text(1932, y, name, ha="right", va="center", color=c,
                fontsize=9.5 if compact else 10.5)
    ax.set_xlim(1932 if not compact else 1932, 2026)
    ax.set_ylim(-0.7, len(ROWS) - 0.15)
    ax.set_yticks([])
    ax.set_xticks([1940, 1960, 1980, 2000, 2020] if not compact
                  else [1940, 1980, 2020])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def figure_timeline():
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.09)
    draw_timeline(fig, ax)
    out = [save_svg(fig, "f7-timeline.svg")]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.96, bottom=0.08)
    draw_timeline(fig, ax, compact=True)
    out.append(save_svg(fig, "f7-timeline-tall.svg"))
    return out


def main():
    os.makedirs(IMG, exist_ok=True)
    written = []
    for make in (figure_hero, figure_physics, figure_heart, figure_finance,
                 figure_ml, figure_grid, figure_traffic, figure_timeline):
        written += make()
    total = 0
    for p in written:
        size = os.path.getsize(p)
        total += size
        print(f"{size:>9,d}  {os.path.relpath(p, ROOT)}")
    print(f"{total:>9,d}  total bytes")


if __name__ == "__main__":
    main()
