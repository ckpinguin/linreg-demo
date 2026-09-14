"""
Linear regression, step by step.

An interactive walkthrough of exactly what LinearRegression().fit() does for
one input variable: the idea, the error measure, the calculus, and the actual
arithmetic, on the same data that linreg.py uses.

    python linreg_tutor.py                 # Back / Next buttons or the arrow keys
    python linreg_tutor.py --save pages    # export every step as a PNG instead
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")                # no window needed for exporting

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.widgets import AxesWidget, Button, Slider
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- The data: identical to linreg.py ---
rng = np.random.default_rng(42)
X = np.arange(20, 106, 5).reshape(-1, 1)
noise = rng.normal(0, 300, size=X.shape[0])
y = 20 * X.ravel() + 300 + noise
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

order = np.argsort(X_train.ravel())      # sorted by size, only for readable tables
xs, ys = X_train.ravel()[order], y_train[order]
xt, yt = X_test.ravel(), y_test
n = len(xs)

# --- Everything the tutorial computes "by hand" ---
x_mean, y_mean = xs.mean(), ys.mean()
dx, dy = xs - x_mean, ys - y_mean
S_xy, S_xx = (dx * dy).sum(), (dx ** 2).sum()
b_best = S_xy / S_xx
a_best = y_mean - b_best * x_mean
resid = ys - (b_best * xs + a_best)
SSE_best, SST = (resid ** 2).sum(), (dy ** 2).sum()

model = LinearRegression().fit(X_train, y_train)   # only used to confirm our numbers

# --- Look ---
C_TRAIN, C_TEST = "#2f6db3", "#ef8a17"
C_FIT, C_GUESS, C_MEAN = "#c62828", "#7b3fa0", "#444444"
C_POS, C_NEG = "#2e9e5b", "#d64545"
XLIM, YLIM = (15, 110), (0, 2800)
FS_TEXT, FS_NOTE = 11.5, 9.5

# --- Abbreviations: listed under the text of every page that uses them, details in a popup on hover ---
GLOSSARY = [    # (pattern, symbol, meaning, formula, details)
    ("SSE", "SSE", "sum of squared errors, how wrong a line is",
     r"$SSE = \sum (y_i - \hat{y}_i)^2$",
     "Square the miss of every training apartment and add them up. Least squares picks the line "
     "with the smallest SSE. Its unit is CHF², hence the big numbers."),
    ("SST", "SST", r"total sum of squares, the error of always guessing $\bar{y}$",
     r"$SST = \sum (y_i - \bar{y})^2$",
     "The SSE of the laziest prediction: a flat line at the average rent. It measures how much the "
     "rents vary before the size is used at all."),
    (r"R\^2|R²", "R^2", "coefficient of determination, the share of the variation explained",
     r"$R^2 = 1 - SSE\,/\,SST$",
     "1 means every dot lies exactly on the line, 0 means no better than always guessing the "
     "average. On test data it can even drop below 0."),
    ("RMSE", "RMSE", "root mean squared error, the typical miss in CHF",
     r"$RMSE = \sqrt{SSE\,/\,n}$",
     "Average the squared errors, then take the square root to get from CHF² back to CHF. "
     "Big misses weigh more than in a plain average of the misses."),
    (r"S_\{xy\}", "S_{xy}", "sum of cross products, do size and rent rise together?",
     r"$S_{xy} = \sum (x_i - \bar{x})(y_i - \bar{y})$",
     "The total signed area of the rectangles from the centre to every point. Positive if bigger "
     "apartments tend to cost more, negative if they tend to cost less."),
    (r"S_\{xx\}", "S_{xx}", "sum of squares of $x$, how spread out the sizes are",
     r"$S_{xx} = \sum (x_i - \bar{x})^2$",
     r"The total area of the squares with side $x_i - \bar{x}$. Dividing $S_{xy}$ by it turns an "
     "area into a slope: CHF per m²."),
]


def sse(b, a):
    """Sum of squared errors of the line y = b*x + a on the training data (works on grids too)."""
    b, a = np.asarray(b, float), np.asarray(a, float)
    return ((ys - b[..., None] * xs - a[..., None]) ** 2).sum(axis=-1)


def m(value, digits=0):
    """A number formatted for use inside $...$, thousands separated by thin spaces."""
    if abs(value) < 0.5 * 10 ** -digits:
        value = 0.0                      # avoid printing "-0"
    return f"{value:,.{digits}f}".replace(",", r"\,")


def rent_axes(t, rect, title=None, xlim=XLIM, ylim=YLIM):
    ax = t.axes(rect)
    ax.set(xlim=xlim, ylim=ylim, xlabel="size x (m²)", ylabel="rent y (CHF)")
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, color="#333")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_train(ax, **kw):
    style = dict(s=48, color=C_TRAIN, edgecolor="white", linewidth=0.8, zorder=4, label="training data")
    ax.scatter(xs, ys, **(style | kw))


def plot_test(ax, **kw):
    style = dict(s=52, color=C_TEST, marker="s", edgecolor="white", linewidth=0.8, zorder=4, label="test data")
    ax.scatter(xt, yt, **(style | kw))


def plot_mean_point(ax, lines=True):
    if lines:
        ax.axvline(x_mean, color=C_MEAN, linestyle="--", linewidth=1, zorder=1)
        ax.axhline(y_mean, color=C_MEAN, linestyle="--", linewidth=1, zorder=1)
    ax.scatter([x_mean], [y_mean], s=260, marker="*", color="gold", edgecolor="black", zorder=6,
               label=r"centre $(\bar{x},\ \bar{y})$")


def readout(ax, x=0.02, y=0.97, **kw):
    """An empty text box in the corner of a plot, to be filled by an update function."""
    style = dict(transform=ax.transAxes, va="top", fontsize=11, zorder=10,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bbb", alpha=0.95))
    return ax.text(x, y, "", **(style | kw))


# --- Text panel building blocks: plain str = paragraph ---
class H(str):
    """Heading."""


class M(str):
    """Centred formula."""


class Box(str):
    """Highlighted result."""


_TOKEN = re.compile(r"(?:[^\s$]*\$[^$]*\$)+[^\s$]*|\S+")   # words, never splitting $math$


def wrap(text, width):
    lines = []
    for raw in text.split("\n"):
        indent = "   " if raw.startswith("• ") else ""
        line, length = "", 0
        for token in _TOKEN.findall(raw):
            w = len(re.sub(r"\\[a-zA-Z]+|[{}^_$\\]", "", token))
            if line and length + 1 + w > width:
                lines.append(line)
                line, length = indent + token, len(indent) + w
            elif line:
                line, length = f"{line} {token}", length + 1 + w
            else:
                line, length = token, w
        lines.append(line)
    return "\n".join(lines)


def glossary_in(text):
    """The GLOSSARY entries a text mentions, in order of first appearance."""
    found = [(match.start(), entry) for entry in GLOSSARY
             if (match := re.search(rf"(?<![A-Za-z])(?:{entry[0]})(?![A-Za-z])", text))]
    return [entry for _, entry in sorted(found, key=lambda f: f[0])]


def explain(entries):
    """Popup text for some GLOSSARY entries."""
    return "\n\n".join(wrap(rf"$\mathbf{{{symbol}}}$ {meaning}" f"\n{formula}\n{details}", 56)
                       for _, symbol, meaning, formula, details in entries)


# --- The app ---
STEPS = []


def step(title):
    def register(page):
        STEPS.append((title, page))
        return page
    return register


class Tutor:
    def __init__(self):
        plt.rcParams.update({"toolbar": "None", "keymap.back": [], "keymap.forward": [],
                             "keymap.home": [], "font.size": 10.5})
        self.fig = plt.figure(figsize=(15, 8.5), facecolor="white")
        self.fig.canvas.manager.set_window_title("Linear regression, step by step")
        self.title = self.fig.text(0.04, 0.945, "", fontsize=18, weight="bold", va="center")
        self.counter = self.fig.text(0.975, 0.945, "", fontsize=11, color="gray", ha="right", va="center")
        self.panel = self.fig.add_axes([0.645, 0.12, 0.33, 0.77])
        self.panel.axis("off")

        self.back = Button(self.fig.add_axes([0.645, 0.035, 0.07, 0.05]), "← Back")
        self.next = Button(self.fig.add_axes([0.905, 0.035, 0.07, 0.05]), "Next →")
        self.back.on_clicked(lambda _: self.show(self.index - 1))
        self.next.on_clicked(lambda _: self.show(self.index + 1))
        self.dots_ax = self.fig.add_axes([0.725, 0.035, 0.17, 0.05])
        self.dots_ax.axis("off")
        self.dots_ax.set(xlim=(-0.6, len(STEPS) - 0.4), ylim=(-1, 1))
        self.dots = self.dots_ax.scatter(range(len(STEPS)), np.zeros(len(STEPS)), s=70)

        self.popup = self.fig.text(0, 0, "", fontsize=10.5, linespacing=1.4, multialignment="left", zorder=100, visible=False,
                                   bbox=dict(boxstyle="round,pad=0.7", fc="#fffdf5", ec="#d9a400"))
        self.hoverable, self.hovered = None, None

        self.guess = {"b": 8.0, "a": 900.0, "h": 400.0}   # the reader's own line, shared by all pages
        self.owned, self.slider_for = [], {}
        self.index = 0

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("resize_event", lambda _: self.show(self.index))  # re-flow text

    # navigation
    def show(self, index):
        self.index = int(np.clip(index, 0, len(STEPS) - 1))
        for item in self.owned:
            if isinstance(item, AxesWidget):
                item.disconnect_events()
                item.ax.remove()
            else:
                self.disconnect_handlers(item)
                item.remove()
        self.owned, self.slider_for = [], {}
        self.hoverable, self.hovered = None, None    # collected again on the next mouse move
        self.popup.set_visible(False)
        self.panel.clear()
        self.panel.axis("off")

        title, page = STEPS[self.index]
        self.title.set_text(f"{self.index + 1}.  {title}")
        self.counter.set_text(f"step {self.index + 1} of {len(STEPS)}   (← → keys)")
        self.dots.set_color(["#333" if i == self.index else "#ccc" for i in range(len(STEPS))])
        page(self)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("right", "pagedown", " ", "enter"):
            self.show(self.index + 1)
        elif event.key in ("left", "pageup", "backspace"):
            self.show(self.index - 1)

    def on_click(self, event):
        if event.inaxes is self.dots_ax and event.xdata is not None:
            self.show(round(event.xdata))

    def on_move(self, event):
        """Explain the abbreviations in the text under the mouse in a popup."""
        if self.hoverable is None:               # after drawing, so tick labels exist too
            self.hoverable = [(text, entries) for text in self.fig.findobj(Text)
                              if text is not self.popup and (entries := glossary_in(text.get_text()))]
        text, entries = next(((text, entries) for text, entries in self.hoverable if text.contains(event)[0]),
                             (None, None))
        if text is self.hovered:
            return
        self.hovered = text
        if text is not None:
            x, y = self.fig.transFigure.inverted().transform((event.x, event.y))
            right, top = x > 0.5, y > 0.5        # open towards the middle of the window
            self.popup.set(text=explain(entries), x=x - 0.01 if right else x + 0.01,
                           y=y - 0.02 if top else y + 0.02, ha="right" if right else "left",
                           va="top" if top else "bottom")
        self.popup.set_visible(text is not None)
        self.fig.canvas.draw_idle()

    def disconnect_handlers(self, artist):
        """3D axes connect mouse handlers to the canvas that would outlive artist.remove() and then crash."""
        callbacks = self.fig.canvas.callbacks
        for handlers in list(callbacks.callbacks.values()):
            for cid, ref in list(handlers.items()):
                if getattr(ref(), "__self__", None) is artist:
                    callbacks.disconnect(cid)

    # page building blocks
    def content(self, slider_rows=0):
        """Rectangle for a page's plot, leaving room for sliders underneath."""
        bottom = 0.09 + 0.045 * slider_rows
        return [0.06, bottom, 0.54, 0.88 - bottom]

    def split(self, slider_rows=0, gap=0.08):
        left, bottom, width, height = self.content(slider_rows)
        half = (width - gap) / 2
        return [left, bottom, half, height], [left + half + gap, bottom, half, height]

    def axes(self, rect, **kw):
        ax = self.fig.add_axes(rect, **kw)
        self.owned.append(ax)
        return ax

    def button(self, rect, label, on_click):
        button = Button(self.fig.add_axes(rect), label)
        button.on_clicked(lambda _: on_click())
        self.owned.append(button)
        return button

    def sliders(self, keys, on_change):
        """Sliders for the shared guess ('b' slope, 'a' intercept) below the plot."""
        spec = {"b": (0, 40, "slope b", "%.2f"), "a": (-800, 1800, "intercept a", "%.0f"), "h": (1, 600, "step h", "%.0f")}
        for row, key in enumerate(keys):
            lo, hi, label, fmt = spec[key]
            self.guess[key] = float(np.clip(self.guess[key], lo, hi))
            ax = self.fig.add_axes([0.12, 0.03 + 0.045 * (len(keys) - 1 - row), 0.36, 0.028])
            slider = Slider(ax, label, lo, hi, valinit=self.guess[key], valfmt=fmt, color=C_GUESS)

            def changed(value, key=key):
                self.guess[key] = value
                on_change()
                self.fig.canvas.draw_idle()

            slider.on_changed(changed)
            self.owned.append(slider)
            self.slider_for[key] = slider
        on_change()

    def set_guess(self, **values):
        for key, value in values.items():
            self.slider_for[key].set_val(value)

    def write(self, *blocks):
        """Lay out headings, paragraphs, formulas and boxes top to bottom in the right panel,
        with the abbreviations they use explained at the bottom."""
        renderer = self.fig.canvas.get_renderer()
        frame = self.panel.get_window_extent(renderer)
        px_per_pt = self.fig.dpi / 72
        width = max(30, int(frame.width / px_per_pt / (FS_TEXT * 0.5)))

        floor = 0.0                              # one line per abbreviation, stacked bottom up
        for _, symbol, meaning, *_ in reversed(glossary_in("\n".join(blocks))):
            note = self.panel.text(0, floor, wrap(rf"$\mathbf{{{symbol}}}$ {meaning}", int(width * FS_TEXT / FS_NOTE)),
                                   fontsize=FS_NOTE, color="#555", va="bottom", transform=self.panel.transAxes)
            floor += note.get_window_extent(renderer).height / frame.height + 0.006
        if floor:
            self.panel.plot([0, 1], [floor + 0.008] * 2, color="#ddd", linewidth=0.8, transform=self.panel.transAxes)
            floor += 0.016

        y = 1.0
        for block in blocks:
            pad = 0
            if isinstance(block, H):
                y -= 0.012
                kw = dict(x=0, s=block, fontsize=12.5, weight="bold", color="#1a1a1a")
            elif isinstance(block, M):
                kw = dict(x=0.5, s=block, fontsize=14.5, ha="center")
            elif isinstance(block, Box):
                pad = 0.6 * 13 * px_per_pt / frame.height
                kw = dict(x=0.5, s=block, fontsize=13, ha="center",
                          bbox=dict(boxstyle="round,pad=0.6", fc="#fff4cc", ec="#d9a400"))
            else:
                kw = dict(x=0, s=wrap(block, width), fontsize=FS_TEXT, linespacing=1.4, color="#222")
            text = self.panel.text(y=y - pad, va="top", transform=self.panel.transAxes, **kw)
            y -= text.get_window_extent(renderer).height / frame.height + 2 * pad + 0.022
        if y < floor - 0.02:
            print(f"note: text panel of step {self.index + 1} overflows", file=sys.stderr)


# === The steps ===

@step("The question")
def page_question(t):
    ax = rent_axes(t, t.content(), title="18 apartments: size vs. monthly rent")
    plot_train(ax)
    plot_test(ax, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    t.write(
        H("What we want"),
        "Each dot is one apartment: its size $x$ in m² and its monthly rent $y$ in CHF. "
        "Bigger apartments tend to cost more, but the dots are noisy: no simple rule hits all of them.",
        "Linear regression finds the one straight line that fits the dots best, so we can "
        "predict the rent of an apartment we have never seen.",
        H("Training and test data"),
        "Exactly like linreg.py, 30% of the apartments are put aside first (orange squares). "
        "Until step 14 we only use the 12 blue training dots. The test dots come back at the "
        "end to check the line honestly.",
        Box(r"$n = 12$ training pairs $(x_i,\ y_i)$"),
        "Use Next / Back or the arrow keys. Some steps have sliders: play with them!",
        "Abbreviations are explained at the bottom of the page. Hover over them, or over any text "
        "that uses them, for more details.",
    )


@step("A straight line has two knobs")
def page_line(t):
    ax = rent_axes(t, t.content(slider_rows=2), xlim=(0, 110), ylim=(-900, 2900), title="Your line")
    ax.axhline(0, color="#999", linewidth=0.8)
    plot_train(ax)
    grid = np.array([0, 110])
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5)
    knob, = ax.plot([], [], "o", color=C_GUESS, markersize=9, zorder=5)
    knob_label = ax.text(0, 0, "", color=C_GUESS, fontsize=11, weight="bold", va="top")
    triangle, = ax.plot([], [], color=C_GUESS, linewidth=1.5, linestyle=":")
    rise_label = ax.text(0, 0, "", color=C_GUESS, fontsize=10.5, ha="center", va="bottom")
    info = readout(ax, fontsize=13)

    def update():
        b, a = t.guess["b"], t.guess["a"]
        line.set_ydata(b * grid + a)
        knob.set_data([0], [a])
        knob_label.set_position((1.5, a - 80))
        knob_label.set_text(f"a = {a:.0f}")
        x0, y0 = 5, 5 * b + a              # left of the data, where nothing is in the way
        triangle.set_data([x0, x0 + 20, x0 + 20], [y0, y0, y0 + 20 * b])
        rise_label.set_position((x0 + 10, y0 + 20 * b + 60))
        rise_label.set_text(f"+20 m²  →  +{20 * b:.0f} CHF")
        info.set_text(rf"$\hat{{y}} = {b:.2f} \cdot x + ({a:.0f})$")

    t.sliders(["b", "a"], update)
    t.write(
        M(r"$\hat{y} = b \cdot x + a$"),
        r"$\hat{y}$ (say: y-hat) is the rent the line predicts for an apartment of size $x$.",
        H("The two knobs"),
        "• $b$, the slope: how many CHF the rent goes up for every extra m². The dotted "
        "triangle shows it: 20 m² to the right, $20 \\cdot b$ CHF up.",
        "• $a$, the intercept: the height where the line crosses $x = 0$. A 0 m² apartment "
        "makes no sense, but $a$ shifts the whole line up and down.",
        H("Try it"),
        "Drag the sliders and lay the line through the dots by eye. Every pair ($b$, $a$) is a "
        "different line. Which one is the best? To answer that, we first need a number that "
        "says how wrong a line is.",
    )


@step("Measure the errors")
def page_residuals(t):
    ax = rent_axes(t, t.content(slider_rows=2), title="Residuals of your line")
    plot_train(ax)
    grid = np.array(XLIM)
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5)
    bars = [ax.plot([xi, xi], [yi, yi], linewidth=2.2)[0] for xi, yi in zip(xs, ys)]
    labels = [ax.text(xi + 1, yi, "", fontsize=8.5, va="center") for xi, yi in zip(xs, ys)]
    info = readout(ax, fontsize=13)

    def update():
        b, a = t.guess["b"], t.guess["a"]
        line.set_ydata(b * grid + a)
        e = ys - (b * xs + a)
        for bar, label, xi, yi, ei in zip(bars, labels, xs, ys, e):
            color = C_POS if ei >= 0 else C_NEG
            bar.set_ydata([yi, yi - ei])
            bar.set_color(color)
            label.set_position((xi + 1, yi - ei / 2))
            label.set_text(f"{ei:+.0f}")
            label.set_color(color)
        info.set_text(rf"$\sum e_i = {m(e.sum())}$ CHF")

    t.sliders(["b", "a"], update)
    t.button([0.525, 0.03, 0.08, 0.073], "flat line at\naverage rent", lambda: t.set_guess(b=0, a=y_mean))
    t.write(
        M(r"$e_i = y_i - \hat{y}_i = y_i - (b\,x_i + a)$"),
        "For every apartment the residual $e_i$ is the vertical gap between its real rent and "
        "the line: positive (green) if the dot lies above the line, negative (red) if below.",
        "Vertical, because we predict $y$ from $x$: the error is measured in CHF.",
        H("First idea: just add the errors up?"),
        r"Click 'flat line at average rent'. It sets $b = 0$ and $a = \bar{y}$, the average rent. "
        "A bad line, it ignores the size completely. Yet the sum of its errors is exactly 0: "
        "the positive and negative errors cancel out.",
        "So the plain sum can't tell a good line from a bad one. We need errors that can't cancel.",
    )


@step("Square the errors")
def page_squares(t):
    ax = rent_axes(t, t.content(slider_rows=2), title="Each squared error drawn as a real square")
    plot_train(ax)
    grid = np.array(XLIM)
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5)
    squares = [ax.add_patch(Rectangle((0, 0), 0, 0, alpha=0.25)) for _ in xs]
    info = readout(ax, family="monospace")

    def update():
        b, a = t.guess["b"], t.guess["a"]
        line.set_ydata(b * grid + a)
        e = ys - (b * xs + a)
        # 1 CHF up and 1 m² across have different lengths on screen; convert so squares look square
        frame = ax.get_window_extent()
        x_per_y = (np.ptp(XLIM) / frame.width) / (np.ptp(YLIM) / frame.height)
        for square, xi, yi, ei in zip(squares, xs, ys, e):
            square.set_bounds(xi, yi - ei, abs(ei) * x_per_y, ei)
            square.set_color(C_POS if ei >= 0 else C_NEG)
        total = (e ** 2).sum()
        verdict = "\n✓ within 5% of the best line!" if total < 1.05 * SSE_best else ""
        info.set_text(f"your SSE {total / 1e6:8.3f} million CHF²\n"
                      f"best SSE {SSE_best / 1e6:8.3f} million CHF²\n"
                      f"you are  {100 * (total / SSE_best - 1):7.1f} % above the best{verdict}")

    t.sliders(["b", "a"], update)
    t.write(
        M(r"$SSE = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - b\,x_i - a)^2$"),
        "SSE, the sum of squared errors. Squaring solves the cancelling problem: a square is "
        "never negative. It also punishes big misses more than small ones. An error twice as "
        "large costs four times as much.",
        "On the plot every squared error is drawn as a real square on its residual. "
        "The SSE is simply their total area.",
        Box("Least squares: pick the line with the smallest total area"),
        H("Your turn"),
        "Drag the sliders and shrink the squares. How close to the best SSE can you get? "
        "Notice how the two knobs fight each other: every change of the slope forces you to "
        "fix the intercept again.",
    )


B_GRID, A_GRID = np.meshgrid(np.linspace(0, 40, 81), np.linspace(-800, 1800, 81))
Z_CAP = 25      # million CHF²; the landscape is cut off above this height


@step("The error landscape")
def page_landscape(t):
    left, bottom, _, height = t.content(slider_rows=2)
    ax = rent_axes(t, [left, bottom, 0.19, height], title="Your line")
    plot_train(ax, s=30)
    grid = np.array(XLIM)
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5)
    info = readout(ax, fontsize=9.5)

    ax3 = t.axes([0.25, bottom - 0.03, 0.35, height + 0.07], projection="3d", computed_zorder=False)
    Z = sse(B_GRID, A_GRID) / 1e6
    ax3.plot_surface(B_GRID, A_GRID, np.where(Z <= Z_CAP, Z, np.nan), cmap="viridis",
                     vmin=0, vmax=Z_CAP, alpha=0.75, linewidth=0, zorder=1)
    ax3.scatter([b_best], [a_best], [SSE_best / 1e6], marker="*", s=350, color="gold",
                edgecolor="black", depthshade=False, zorder=5)
    dot, = ax3.plot([], [], [], "o", color=C_GUESS, markersize=10, markeredgecolor="white", zorder=6)
    stem, = ax3.plot([], [], [], color=C_GUESS, linewidth=1.5, zorder=6)
    ax3.set(xlabel="slope b", ylabel="intercept a", zlabel="SSE (millions)", zlim=(0, Z_CAP))
    ax3.view_init(elev=30, azim=-60)

    def update():
        b, a = t.guess["b"], t.guess["a"]
        line.set_ydata(b * grid + a)
        z = sse(b, a) / 1e6
        dot.set_data_3d([b], [a], [min(z, Z_CAP)])
        stem.set_data_3d([b, b], [a, a], [0, min(z, Z_CAP)])
        info.set_text(f"SSE = {z:.2f}\nmillion CHF²")

    t.sliders(["b", "a"], update)
    t.write(
        "Every line is a pair of numbers ($b$, $a$): a point on a map. Above every point we "
        "raise the SSE of that line. The result is a landscape, a long curved valley.",
        "The purple dot is your line from the sliders. The gold star marks the lowest point of "
        "the valley: the best line. (Drag the 3D plot to rotate it.)",
        H("How do we find the bottom?"),
        "You could try lines at random, or walk downhill in small steps. That is called gradient "
        "descent, and it is how neural networks learn.",
        "For a straight line there is a shortcut. At the very bottom the ground is flat in every "
        "direction. Flat means the derivative (the steepness of the ground) is zero, and that "
        "equation we can solve exactly. Next step: how a derivative is calculated.",
    )


@step("Derivatives: how steep is a curve?")
def page_derivative(t):
    xk, yk, b = xs[-1], ys[-1], 20                 # the largest apartment, slope frozen at a round number

    def f(a):
        return (yk - b * xk - a) ** 2

    left, right = t.split(slider_rows=2)
    ax = rent_axes(t, left, ylim=(0, 4000), title=f"One apartment and a line with slope {b}")
    plot_train(ax, alpha=0.25, label=None)
    ax.scatter([xk], [yk], s=80, color=C_TRAIN, edgecolor="black", zorder=5)
    grid = np.array(XLIM)
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5)
    gap, = ax.plot([xk, xk], [yk, yk], color=C_NEG, linewidth=2.5)
    gap_label = ax.text(xk - 2, yk, "", ha="right", va="center", fontsize=12, color=C_NEG, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))

    ax2 = t.axes(right)
    ax2.set(xlabel="intercept a", ylabel="f(a) (million CHF²)", xlim=(A_LINE[0], A_LINE[-1]),
            ylim=(0, f(A_LINE).max() / 1e6 * 1.05))
    ax2.set_title(r"Its squared error $f(a) = u^2$", loc="left", fontsize=11.5, color="#333")
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.plot(A_LINE, f(A_LINE) / 1e6, color="#555", linewidth=2)
    secant, = ax2.plot([], [], color=C_TEST, linewidth=2, label="secant through a and a + h")
    run_rise, = ax2.plot([], [], color=C_TEST, linewidth=1.2, linestyle=":")
    ends, = ax2.plot([], [], "o", color=C_TEST, markersize=8, zorder=5)
    tangent, = ax2.plot([], [], color=C_GUESS, linewidth=2, label="tangent at a")
    dot, = ax2.plot([], [], "o", color=C_GUESS, markersize=9, zorder=6)
    ax2.legend(loc="center left", frameon=False)
    info = readout(ax2, family="monospace")

    def update():
        a, h = t.guess["a"], t.guess["h"]
        u = yk - b * xk - a
        line.set_ydata(b * grid + a)
        gap.set_ydata([yk, yk - u])
        gap_label.set_position((xk - 2, yk - u / 2))
        gap_label.set_text(f"u = {u:+.0f}")
        rise = (f(a + h) - f(a)) / h
        span = np.array([a - 500, a + h + 500])
        secant.set_data(span, (f(a) + rise * (span - a)) / 1e6)
        run_rise.set_data([a, a + h, a + h], np.array([f(a), f(a), f(a + h)]) / 1e6)
        ends.set_data([a, a + h], np.array([f(a), f(a + h)]) / 1e6)
        tangent.set_data(span, (f(a) - 2 * u * (span - a)) / 1e6)
        dot.set_data([a], [f(a) / 1e6])
        info.set_text(f"secant slope  {rise:+6.0f}\n"
                      f"tangent slope {-2 * u:+6.0f}\n"
                      f"difference    {rise + 2 * u:+6.0f} = h")

    t.sliders(["a", "h"], update)
    t.write(
        "A derivative is the steepness of a curve at one point. Example: the squared error of the "
        "largest apartment, as a function of the intercept $a$.",
        M(rf"$f(a) = u^2$   with   $u = y - {b}\,x - a$"),
        H("1. Rise over run"),
        "Move $a$ by a step $h$, so $u$ shrinks by $h$. The slope of the secant (orange):",
        M(r"$\dfrac{f(a+h) - f(a)}{h} = \dfrac{(u-h)^2 - u^2}{h} = -2u + h$"),
        H("2. Let h shrink to zero"),
        "Drag $h$ to the left: the secant becomes the tangent and the leftover $h$ vanishes:",
        Box(r"$f\,'(a) = -2u = -2\,(y - b\,x - a)$"),
        H("Rules that save the work"),
        r"• Chain rule: $(u^2)' = 2u \cdot u'$, with $u' = -1$ for $a$ and $u' = -x$ for $b$.",
        "• Sum rule: the derivative of the SSE is the sum over all 12 squared errors.",
        r"• Partial $\partial$: move one knob, treat the other as a fixed number.",
    )


def tangent_plot(t, rect, xlabel, title):
    """Right-hand plot for the calculus steps: an SSE curve, a tangent and the lowest point."""
    ax = t.axes(rect)
    ax.set(xlabel=xlabel, ylabel="SSE (million CHF²)")
    ax.set_title(title, loc="left", fontsize=11.5, color="#333")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    curve, = ax.plot([], [], color="#555", linewidth=2)
    tangent, = ax.plot([], [], color=C_GUESS, linewidth=2)
    dot, = ax.plot([], [], "o", color=C_GUESS, markersize=9, zorder=5)
    bottom, = ax.plot([], [], "*", color="gold", markeredgecolor="black", markersize=17, zorder=4)
    return ax, curve, tangent, dot, bottom, readout(ax, x=0.5, ha="center", fontsize=12.5)


def update_tangent(parts, grid, values, at, value, derivative, variable):
    ax, curve, tangent, dot, bottom, info = parts
    curve.set_data(grid, values / 1e6)
    bottom.set_data([grid[values.argmin()]], [values.min() / 1e6])
    half = np.ptp(grid) / 8
    span = np.array([at - half, at + half])
    tangent.set_data(span, (value + derivative * (span - at)) / 1e6)
    dot.set_data([at], [value / 1e6])
    ax.set(xlim=(grid[0], grid[-1]), ylim=(0, values.max() / 1e6 * 1.05))
    direction = "→ increase " if derivative < 0 else "→ decrease "
    hint = "flat: this is the bottom!" if abs(value / values.min() - 1) < 0.005 else direction + variable
    info.set_text(rf"$\partial\, SSE / \partial {variable} = {m(derivative)}$" f"\n{hint}")


A_LINE = np.linspace(-800, 1800, 261)
B_LINE = np.linspace(0, 40, 201)


@step("Calculus I: the best intercept a")
def page_calculus_a(t):
    left, right = t.split(slider_rows=2)
    ax = rent_axes(t, left, title="Your line")
    plot_train(ax)
    plot_mean_point(ax, lines=False)
    grid = np.array(XLIM)
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5, label="your line")
    best, = ax.plot(grid, grid * 0, color=C_FIT, linewidth=1.5, linestyle="--", label="best a for this b")
    ax.legend(loc="lower right", frameon=False)
    parts = tangent_plot(t, right, "intercept a", "SSE when only a changes (b frozen)")

    def update():
        b, a = t.guess["b"], t.guess["a"]
        line.set_ydata(b * grid + a)
        best.set_ydata(b * grid + y_mean - b * x_mean)
        derivative = -2 * (ys - b * xs - a).sum()
        update_tangent(parts, A_LINE, sse(b, A_LINE), a, sse(b, a), derivative, "a")

    t.sliders(["b", "a"], update)
    t.write(
        H("Freeze the slope, change only a"),
        "The right plot shows the SSE for every $a$. It is a parabola, and its lowest point is "
        "where the tangent (purple) is flat. The steepness of the tangent is the derivative.",
        "Every squared error has the derivative $-2u$ from step 6, with $u = y_i - b\\,x_i - a$. "
        "Add them up over all apartments (sum rule) and set the result to zero:",
        M(r"$\dfrac{\partial\, SSE}{\partial a} = -2 \sum (y_i - b\,x_i - a) = 0$"),
        "Divide by −2 and split the sum. Adding $a$ up $n$ times gives $n\\,a$:",
        M(r"$\sum y_i - b \sum x_i - n\,a = 0$"),
        "Solve for $a$ and divide by $n$. The sums turn into averages:",
        Box(r"$a = \bar{y} - b\,\bar{x}$"),
        r"Result 1: whatever the slope, the best line passes through the centre of the data, "
        r"the point $(\bar{x},\ \bar{y})$ (gold star). Move $b$: the dashed line always "
        "pivots around the star.",
    )


@step("Calculus II: the best slope b")
def page_calculus_b(t):
    left, right = t.split(slider_rows=1)
    ax = rent_axes(t, left, title=r"Line through the centre with slope b")
    plot_train(ax)
    plot_mean_point(ax, lines=False)
    grid = np.array(XLIM)
    line, = ax.plot(grid, grid * 0, color=C_GUESS, linewidth=2.5)
    parts = tangent_plot(t, right, "slope b", r"SSE when $a = \bar{y} - b\,\bar{x}$")
    valley = sse(B_LINE, y_mean - B_LINE * x_mean)

    def update():
        b = t.guess["b"]
        line.set_ydata(y_mean + b * (grid - x_mean))
        derivative = -2 * (dx * (dy - b * dx)).sum()
        update_tangent(parts, B_LINE, valley, b, sse(b, y_mean - b * x_mean), derivative, "b")

    t.sliders(["b"], update)
    t.write(
        r"Plug result 1 into the SSE: replace $a$ by $\bar{y} - b\,\bar{x}$. Now the line always "
        "pivots around the centre, and only $b$ is left:",
        M(r"$SSE(b) = \sum\ ((y_i - \bar{y}) - b\,(x_i - \bar{x}))^2$"),
        r"Derivative equal to zero again. In the chain rule $2u \cdot u'$ the part being squared is "
        r"$u = (y_i - \bar{y}) - b\,(x_i - \bar{x})$, with inner derivative $u' = -(x_i - \bar{x})$:",
        M(r"$-2 \sum (x_i - \bar{x})\,((y_i - \bar{y}) - b\,(x_i - \bar{x})) = 0$"),
        "Multiply out and bring the $b$ term to the other side:",
        M(r"$\sum (x_i-\bar{x})(y_i-\bar{y}) = b \sum (x_i-\bar{x})^2$"),
        Box(r"$b = \dfrac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2} = \dfrac{S_{xy}}{S_{xx}}$"),
        "Result 2. Results 1 and 2 give the exact bottom of the valley: no searching, "
        "no guessing. Next, we compute them with the real numbers.",
    )


@step("Compute 1: the means")
def page_means(t):
    ax = rent_axes(t, t.content(), title="The centre of the training data")
    plot_train(ax)
    plot_mean_point(ax)
    ax.annotate(rf"$\bar{{x}} = {x_mean:.2f}$", (x_mean, 0), xytext=(6, 8), textcoords="offset points",
                fontsize=13, color=C_MEAN)
    ax.annotate(rf"$\bar{{y}} = {y_mean:.2f}$", (XLIM[0], y_mean), xytext=(6, 8), textcoords="offset points",
                fontsize=13, color=C_MEAN)
    ax.legend(loc="upper left", frameon=False)
    t.write(
        H("Average size and average rent"),
        "sizes $x_i$:  " + ", ".join(f"{v:.0f}" for v in xs),
        "rents $y_i$:  " + ", ".join(f"{v:.0f}" for v in ys),
        M(rf"$\bar{{x}} = \dfrac{{1}}{{n}} \sum x_i = \dfrac{{{m(xs.sum())}}}{{12}} = {m(x_mean, 2)}$ m²"),
        M(rf"$\bar{{y}} = \dfrac{{1}}{{n}} \sum y_i = \dfrac{{{m(ys.sum(), 1)}}}{{12}} = {m(y_mean, 2)}$ CHF"),
        r"The gold star at $(\bar{x},\ \bar{y})$ is the centre of the data. "
        "By result 1, the best line will pass right through it.",
        "The dashed lines split the plot into four quadrants around the centre. "
        "Next we measure every apartment relative to this centre.",
    )


@step("Compute 2: deviations become rectangles")
def page_rectangles(t):
    ax = rent_axes(t, t.content(), title="Rectangles from the centre to every point")
    for xi, yi, dxi, dyi in zip(xs, ys, dx, dy):
        color = C_POS if dxi * dyi >= 0 else C_NEG
        ax.add_patch(Rectangle((x_mean, y_mean), dxi, dyi, facecolor=color, edgecolor=color,
                               alpha=0.16, linewidth=1.2))
    plot_train(ax)
    plot_mean_point(ax)
    i = 0                                  # annotate the smallest apartment as an example
    ax.annotate(rf"$x - \bar{{x}} = {dx[i]:.2f}$" "\n" rf"$y - \bar{{y}} = {dy[i]:.1f}$" "\n"
                rf"area $= {m(dx[i] * dy[i])}$",
                (xs[i], ys[i]), xytext=(-10, -75), textcoords="offset points", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#555"))
    for (qx, qy, label, color) in [(0.97, 0.97, "both +  →  area +", C_POS), (0.03, 0.03, "both −  →  area +", C_POS),
                                   (0.03, 0.97, "x −, y +  →  area −", C_NEG), (0.97, 0.03, "x +, y −  →  area −", C_NEG)]:
        ax.text(qx, qy, label, transform=ax.transAxes, color=color, fontsize=11, weight="bold",
                ha="right" if qx > 0.5 else "left", va="top" if qy > 0.5 else "bottom")
    t.write(
        r"Measure every apartment from the centre: $x_i - \bar{x}$ (right +, left −) and "
        r"$y_i - \bar{y}$ (up +, down −). Their product is the signed area of the rectangle "
        "between the point and the star.",
        "• Upper right and lower left: both deviations have the same sign, so the area is "
        "positive (green). These apartments say: bigger means more expensive.",
        "• Upper left and lower right: opposite signs, negative area (red). These apartments disagree.",
        M(rf"$S_{{xy}} = \sum (x_i-\bar{{x}})(y_i-\bar{{y}}) = {m(S_xy, 1)}$"),
        "Almost everything here is green, so the slope will be clearly positive.",
        "$S_{xx}$ is the same idea with $x$ alone: the total area of squares with side "
        r"$x_i - \bar{x}$. It measures how spread out the sizes are.",
        M(rf"$S_{{xx}} = \sum (x_i-\bar{{x}})^2 = {m(S_xx, 2)}$"),
    )


@step("Compute 3: the worksheet")
def page_worksheet(t):
    ax = t.axes(t.content())
    ax.axis("off")
    header = [r"$x_i$", r"$y_i$", r"$x_i-\bar{x}$", r"$y_i-\bar{y}$",
              r"$(x_i-\bar{x})(y_i-\bar{y})$", r"$(x_i-\bar{x})^2$"]
    def cell(value, spec):
        return f"{value:{spec}}".replace(",", " ").replace("-", "−")

    rows = [[cell(a, ".0f"), cell(b, ",.1f"), cell(c, "+.2f"), cell(d, "+,.1f"), cell(e, "+,.1f"), cell(f, ",.2f")]
            for a, b, c, d, e, f in zip(xs, ys, dx, dy, dx * dy, dx ** 2)]
    rows.append(["Σ " + cell(xs.sum(), ".0f"), "Σ " + cell(ys.sum(), ",.1f"), "0", "0",
                 cell(S_xy, ",.1f"), cell(S_xx, ",.2f")])
    table = ax.table(cellText=rows, colLabels=header, bbox=[0, 0.02, 1, 0.96], cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#ddd")
        if row == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(ha="center")
        elif col in (4, 5):
            cell.set_facecolor("#fbf6e3")
        if row == len(rows):
            cell.set_facecolor("#fff4cc")
            cell.set_text_props(weight="bold")
    ax.set_title("All 12 training apartments (deviations always sum to 0)", loc="left", fontsize=11.5, color="#333")
    t.write(
        H("Slope"),
        M(rf"$b = \dfrac{{S_{{xy}}}}{{S_{{xx}}}} = \dfrac{{{m(S_xy, 1)}}}{{{m(S_xx, 2)}}} = {m(b_best, 4)}$"),
        f"Every extra m² adds about {b_best:.2f} CHF of rent.",
        H("Intercept"),
        M(rf"$a = \bar{{y}} - b\,\bar{{x}} = {m(y_mean, 2)} - {m(b_best, 4)} \cdot {m(x_mean, 2)}$"),
        M(rf"$a = {m(a_best, 2)}$"),
        Box(rf"$\hat{{y}} = {b_best:.2f}\,x + {a_best:.2f}$"),
        H("Check against scikit-learn"),
        f"LinearRegression().fit() in linreg.py finds ${m(model.coef_[0], 4)}$ for the slope and "
        f"${m(model.intercept_, 4)}$ for the intercept. Exactly our numbers: fit() solves these "
        "same equations.",
    )


@step("The best line")
def page_best_line(t):
    left, right = t.split(gap=0.07)
    ax = rent_axes(t, left, title="Least-squares line and its residuals")
    grid = np.array(XLIM)
    ax.plot(grid, t.guess["b"] * grid + t.guess["a"], color=C_GUESS, linewidth=1.5, alpha=0.5, label="your last guess")
    for xi, yi, ei in zip(xs, ys, resid):
        ax.plot([xi, xi], [yi, yi - ei], color=C_POS if ei >= 0 else C_NEG, linewidth=2)
    ax.plot(grid, b_best * grid + a_best, color=C_FIT, linewidth=2.5, label="best line")
    plot_train(ax)
    plot_mean_point(ax, lines=False)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    ax2 = t.axes(right)
    Z = sse(B_GRID, A_GRID)
    levels = np.geomspace(SSE_best, Z.max(), 16)          # geometric steps show the deep valley floor
    ax2.contourf(B_GRID, A_GRID, Z, levels=levels, cmap="viridis")
    ax2.contour(B_GRID, A_GRID, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.5)
    ax2.plot(B_LINE, y_mean - B_LINE * x_mean, color="white", linestyle="--", linewidth=1.2)
    ax2.plot([t.guess["b"]], [t.guess["a"]], "o", color=C_GUESS, markersize=9, markeredgecolor="white")
    ax2.plot([b_best], [a_best], "*", color="gold", markeredgecolor="black", markersize=18)
    ax2.set(xlabel="slope b", ylabel="intercept a", xlim=(0, 40), ylim=(-800, 1800))
    ax2.set_title("Error map seen from above (dark = low SSE)", loc="left", fontsize=11.5, color="#333")
    t.write(
        Box(rf"$\hat{{y}} = {b_best:.2f}\,x + {a_best:.2f}$"),
        H("Checks you can do yourself"),
        rf"• It passes through the centre: ${b_best:.4f} \cdot {x_mean:.2f} + {a_best:.2f} = {y_mean:.2f}$",
        rf"• Result 1 holds, the residuals add up to zero: $\sum e_i = {m(resid.sum(), 2)}$",
        rf"• Result 2 holds too: $\sum (x_i-\bar{{x}})\,e_i = {m((dx * resid).sum(), 2)}$",
        f"• No other line has a smaller SSE than {SSE_best / 1e6:.3f} million CHF². On the error "
        "map the formulas land exactly at the bottom of the valley (star). The purple dot is "
        "your last guess from the sliders.",
        H("Why the valley is long and tilted"),
        "Tilting the line also moves it a lot far away at $x = 0$, so the intercept has to "
        r"follow: $a = \bar{y} - b\,\bar{x}$ (white dashed line). That is why the two knobs "
        "fought each other in step 4.",
    )


@step("How good is the line? R²")
def page_r2(t):
    rmse_train = np.sqrt(SSE_best / n)
    left, bottom, width, _ = t.content()
    top_left, top_right = t.split(gap=0.06)
    for rect, title, fitted, color in [(top_left, r"Without the line: always guess $\bar{y}$", np.full(n, y_mean), "#8a8a8a"),
                                       (top_right, "With the regression line", b_best * xs + a_best, C_FIT)]:
        rect = [rect[0], 0.36, rect[2], 0.52]
        ax = rent_axes(t, rect, title=title)
        for xi, yi, fi in zip(xs, ys, fitted):
            ax.plot([xi, xi], [yi, fi], color=color, linewidth=2, alpha=0.8)
        ax.plot(xs[[0, -1]], fitted[[0, -1]], color=color, linewidth=2.5)
        plot_train(ax)

    bars = t.axes([left + 0.07, bottom, width - 0.07, 0.15])
    bars.barh([1, 0], [SST / 1e6, SSE_best / 1e6], color=["#8a8a8a", C_FIT], height=0.6)
    bars.set_yticks([1, 0], ["SST (average)", "SSE (line)"])
    bars.set_xlabel("total squared error (million CHF²)")
    bars.spines[["top", "right"]].set_visible(False)
    bars.text(SSE_best / 1e6, 0, f"  only {100 * SSE_best / SST:.0f}% of the error is left",
              va="center", fontsize=11)
    t.write(
        "Is the line any good? Compare it with the laziest possible prediction: ignore the size "
        r"and always guess the average rent $\bar{y}$.",
        M(rf"$SST = \sum (y_i - \bar{{y}})^2 = {m(SST / 1e6, 3)}$ million"),
        "With the line, this much squared error is left over:",
        M(rf"$SSE = \sum (y_i - \hat{{y}}_i)^2 = {m(SSE_best / 1e6, 3)}$ million"),
        Box(rf"$R^2 = 1 - \dfrac{{SSE}}{{SST}} = {1 - SSE_best / SST:.3f}$"),
        "R² is the share of the rent differences the line explains. 0 means no better than the "
        "average, 1 means every dot lies exactly on the line.",
        M(rf"$RMSE = \sqrt{{SSE / n}} = {rmse_train:.1f}$ CHF"),
        "RMSE is the typical miss in CHF. Careful: these are training numbers, measured on the "
        "same dots the line was fitted to, so they are a bit too optimistic.",
    )


@step("The honest test")
def page_test(t):
    pred_test = b_best * xt + a_best
    r2_test = 1 - ((yt - pred_test) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    rmse_test = np.sqrt(((yt - pred_test) ** 2).mean())
    rent_48 = b_best * 48 + a_best

    ax = rent_axes(t, t.content(), title="Unseen apartments")
    grid = np.array(XLIM)
    ax.plot(grid, b_best * grid + a_best, color=C_FIT, linewidth=2.5, label="best line (from training)")
    for xi, yi, pi in zip(xt, yt, pred_test):
        ax.plot([xi, xi], [yi, pi], color=C_TEST, linewidth=2)
    plot_train(ax, alpha=0.3)
    plot_test(ax)
    ax.plot([48, 48, XLIM[0]], [0, rent_48, rent_48], color="#333", linestyle=":", linewidth=1.5)
    ax.plot([48], [rent_48], "D", color="#333", markersize=9, zorder=6, label="new apartment: 48 m²")
    ax.annotate(f"48 m²  →  {rent_48:.0f} CHF", (48, rent_48), xytext=(12, -28), textcoords="offset points",
                fontsize=12, weight="bold")
    ax.legend(loc="upper left", frameon=False)
    t.write(
        "The 6 test apartments come back. The line has never seen them, so its errors on them "
        "are an honest preview of how it will do on new apartments.",
        M(rf"$R^2_{{test}} = {r2_test:.3f}$      $RMSE_{{test}} = {rmse_test:.2f}$ CHF"),
        "Same formulas as in step 13, and exactly the numbers linreg.py prints. "
        f"Training RMSE was {np.sqrt(SSE_best / n):.0f} CHF. If the test error were far worse, "
        "the model would have memorised noise instead of learning the trend (overfitting). "
        "A line with just 2 numbers can hardly do that.",
        H("Predict a new apartment"),
        M(rf"$\hat{{y}} = {b_best:.2f} \cdot 48 + {a_best:.2f} = {rent_48:.0f}$ CHF"),
        "That is all a trained linear regression is: two numbers, used in one formula.",
    )


@step("Summary: the whole recipe")
def page_summary(t):
    ax = rent_axes(t, t.content(), title="Linear regression on the apartment data")
    grid = np.array(XLIM)
    ax.plot(grid, b_best * grid + a_best, color=C_FIT, linewidth=2.5,
            label=rf"$\hat{{y}} = {b_best:.2f}\,x + {a_best:.2f}$")
    plot_train(ax)
    plot_test(ax)
    plot_mean_point(ax, lines=False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    t.write(
        H("Least squares in 6 steps"),
        r"1.  Collect $n$ pairs $(x_i,\ y_i)$ and put some aside for testing.",
        r"2.  Means: $\bar{x}$ and $\bar{y}$",
        r"3.  $S_{xy} = \sum (x_i-\bar{x})(y_i-\bar{y})$  and  $S_{xx} = \sum (x_i-\bar{x})^2$",
        r"4.  Slope $b = S_{xy} / S_{xx}$,  intercept $a = \bar{y} - b\,\bar{x}$",
        r"5.  Predict: $\hat{y} = b\,x + a$",
        r"6.  Evaluate on the test data: $R^2 = 1 - SSE/SST$,  $RMSE = \sqrt{SSE/n}$",
        H("Why it works"),
        "SSE measures how wrong a line is. Its landscape over ($b$, $a$) is a bowl, and the "
        "slope and intercept formulas are exactly where both derivatives are zero: the bottom "
        "of the bowl.",
        H("What comes next"),
        "With more inputs (size, rooms, floor, ...) the same idea is written with matrices: "
        r"$\hat{\beta} = (X^T X)^{-1} X^T y$. Still the same goal: the smallest sum of squared errors.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", metavar="DIR", help="save every step as a PNG into DIR instead of opening a window")
    args = parser.parse_args()

    tutor = Tutor()
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(len(STEPS)):
            tutor.show(i)
            tutor.fig.savefig(out / f"step_{i + 1:02d}.png")
        print(f"saved {len(STEPS)} steps to {out}")
    else:
        tutor.show(0)
        plt.show()


if __name__ == "__main__":
    main()
