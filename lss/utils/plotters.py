import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import lss.utils.misc as utils
import pyvista as pv


def plot_img(ax, img, normalize=True):
    if isinstance(img, torch.Tensor) and img.shape[0] == 3:
        img = img.permute(1, 2, 0)

    img = utils.to_cpu(img)

    if normalize:
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)

    if img.shape[0] == 1:
        img = img.squeeze()

    ax.imshow(img, interpolation="nearest", cmap="summer")
    ax.axis("off")


def add_label(ax, label):
    if label is None:
        return

    ax.text(
        0.5,
        0.98,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        color="white",
        bbox=dict(facecolor="black", alpha=1, pad=4, edgecolor="none"),
    )


def plot(img, label=None, output_path="tmp.png", save_figure=False):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_position([0, 0, 1, 1])

    plot_img(ax, img)
    add_label(ax, label)
    if not save_figure:
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_out = np.asarray(buf)[:, :, :3]
        plt.close(fig)
        return img_out.transpose(2, 0, 1)
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saved figure in {output_path}")
        fig.savefig(output_path, dpi=100, pad_inches=0)
        plt.close(fig)


def visualize_pcl(points, color=None, save_path=None, show_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    points = np.asarray(points)
    cloud = pv.PolyData(points)
    plotter = pv.Plotter(off_screen=save_path is not None)
    plotter.set_background("white")

    if color is not None:
        if isinstance(color, torch.Tensor):
            color = color.detach().cpu().numpy()
        color = np.asarray(color)
        if color.max() <= 1.0:
            color = (color * 255).astype(np.uint8)
        cloud["colors"] = color
        plotter.add_points(
            cloud,
            scalars="colors",
            rgb=True,
            point_size=3,
            render_points_as_spheres=False,
        )
    else:
        plotter.add_points(
            cloud,
            color="black",
            point_size=3,
            render_points_as_spheres=False,
        )

    # Show coordinate origin
    if show_origin:
        origin = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
        plotter.add_points(
            origin,
            color="red",
            point_size=12,
            render_points_as_spheres=True,
        )

    if save_path is not None:
        plotter.show(auto_close=False)
        plotter.screenshot(save_path)
        plotter.close()
    else:
        plotter.show()
