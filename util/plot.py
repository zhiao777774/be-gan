import matplotlib.pyplot as plt


def clear_patch(ax: plt.Axes, gid: str) -> None:
    # print(ax.patches)
    for c in ax.patches:
        if c.get_gid() == gid:
            c.remove()
            break


def clear_line(ax: plt.Axes, gid: str) -> None:
    for l in ax.lines:
        if l.get_gid() == gid:
            l.remove()
            break


def clear_collection(ax: plt.Axes, gid: str) -> None:
    for c in ax.collections:
        if c.get_gid() == gid:
            c.remove()
