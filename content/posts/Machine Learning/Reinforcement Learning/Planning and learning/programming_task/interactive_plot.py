import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

os.chdir("content/posts/Reinforcement Learning/Planning and learning/programming_task/data")


def _get_series(run, name):
    values = run.get(name)
    if values is None:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def _get_bounds(run, algorithm_names):
    lengths = []
    mins = []
    maxs = []

    for name in algorithm_names:
        series = _get_series(run, name)
        if series.size == 0:
            continue
        lengths.append(series.size)
        mins.append(float(series.min()))
        maxs.append(float(series.max()))

    if not lengths:
        return 1, -1.0, 1.0

    y_min = min(mins)
    y_max = max(maxs)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    return max(lengths), y_min, y_max


def _format_kappa(value):
    return f"{float(value):.2e}"


def interactive_plot(data, title="Planning Steps"):
    steps = np.array(sorted(data))
    if steps.size == 0:
        raise ValueError("data must contain at least one step")

    algorithm_names = sorted({name for step in data.values() for name in step})
    first_step = data[int(steps[0])]
    max_len, y_min, y_max = _get_bounds(first_step, algorithm_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.22)

    lines = {}
    for name in algorithm_names:
        series = _get_series(first_step, name)
        line, = ax.plot(np.arange(series.size), series, label=name, linewidth=2)
        line.set_visible(series.size > 0)
        lines[name] = line

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"{title}: {int(steps[0])}")
    ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    slider_ax = fig.add_axes((0.12, 0.08, 0.76, 0.04))
    slider = Slider(
        ax=slider_ax,
        label=title,
        valmin=0,
        valmax=len(steps) - 1,
        valinit=0,
        valstep=1,
    )

    def update(_):
        step = int(steps[int(slider.val)])
        run = data[step]
        max_len, local_min, local_max = _get_bounds(run, algorithm_names)

        for name in algorithm_names:
            series = _get_series(run, name)
            lines[name].set_data(np.arange(series.size), series)
            lines[name].set_visible(series.size > 0)

        ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
        ax.set_ylim(local_min, local_max)
        ax.set_title(f"{title}: {step}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def interactive_plot_2d(data, kappa_title="Kappa", planning_title="Planning Steps"):
    kappa_keys = np.array(sorted(data))
    if kappa_keys.size == 0:
        raise ValueError("data must contain at least one kappa value")

    first_kappa = float(kappa_keys[0])
    planning_keys = np.array(sorted(data[first_kappa]))
    if planning_keys.size == 0:
        raise ValueError("each kappa slice must contain at least one planning step")

    algorithm_names = sorted(
        {
            name
            for per_kappa in data.values()
            for per_planning in per_kappa.values()
            for name in per_planning
        }
    )

    first_run = data[first_kappa][int(planning_keys[0])]
    max_len, y_min, y_max = _get_bounds(first_run, algorithm_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.30)

    lines = {}
    for name in algorithm_names:
        series = _get_series(first_run, name)
        line, = ax.plot(np.arange(series.size), series, label=name, linewidth=2)
        line.set_visible(series.size > 0)
        lines[name] = line

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(
        f"{kappa_title}: {_format_kappa(first_kappa)}, {planning_title}: {int(planning_keys[0])}"
    )
    ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
    ax.set_ylim(y_min, y_max)
    ax.legend()

    kappa_slider_ax = fig.add_axes((0.12, 0.12, 0.76, 0.04))
    planning_slider_ax = fig.add_axes((0.12, 0.05, 0.76, 0.04))

    kappa_slider = Slider(
        ax=kappa_slider_ax,
        label=kappa_title,
        valmin=0,
        valmax=len(kappa_keys) - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )
    planning_slider = Slider(
        ax=planning_slider_ax,
        label=planning_title,
        valmin=0,
        valmax=len(planning_keys) - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )

    kappa_slider.valtext.set_text(_format_kappa(first_kappa))
    planning_slider.valtext.set_text(str(int(planning_keys[0])))

    def update(_):
        kappa_key = float(kappa_keys[int(kappa_slider.val)])
        planning_key = int(planning_keys[int(planning_slider.val)])
        run = data[kappa_key][planning_key]
        max_len, local_min, local_max = _get_bounds(run, algorithm_names)

        for name in algorithm_names:
            series = _get_series(run, name)
            lines[name].set_data(np.arange(series.size), series)
            lines[name].set_visible(series.size > 0)

        ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
        ax.set_ylim(local_min, local_max)
        ax.set_title(f"{kappa_title}: {_format_kappa(kappa_key)}, {planning_title}: {planning_key}")
        kappa_slider.valtext.set_text(_format_kappa(kappa_key))
        planning_slider.valtext.set_text(str(planning_key))
        fig.canvas.draw_idle()

    kappa_slider.on_changed(update)
    planning_slider.on_changed(update)
    plt.show()


filename = "kappa_planning_study_logspace_planning_1_to_25_6000"
with open(f"{filename}.pickle", "rb") as f:
    data = pickle.load(f)

interactive_plot_2d(data, kappa_title="Kappa", planning_title="Planning Steps")
