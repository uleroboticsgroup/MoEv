import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FIGSIZE = (14, 18)        # vertical y más compacto
TICK_FONTSIZE = 13
ANN_FONTSIZE  = 16
TITLE_FONTSIZE = 15
SAVE_DPI = 300

sns.set_style("white")

cms = {
    # DATASET 1/1000
    (0, 0): (np.array([[525, 361],[218, 668]]), "KNN\n(a)"),
    (0, 1): (np.array([[347, 539],[387, 499]]), "LR\n(b)"),
    (0, 2): (np.array([[ 62, 824],[  3, 883]]), "LSVC\n(c)"),
    (1, 0): (np.array([[704, 182],[163, 723]]), "MLP\n(d)"),
    (1, 1): (np.array([[215, 671],[  0, 886]]), "SGD\n(e)"),
    (1, 2): (np.array([[867,  19],[886,   0]]), "RF\n(f)"),
    (2, 0): (np.array([[388, 498],[194, 692]]), "VC\n(g)"),

    # DATASET 1/500
    (2, 1): (np.array([[1069, 668],[757, 980]]), "KNN\n(h)"),
    (2, 2): (np.array([[770, 967],[873, 864]]), "LR\n(i)"),
    (3, 0): (np.array([[ 861, 876],[  1073, 664]]), "LSVC\n(j)"),
    (3, 1): (np.array([[842, 895],[1355, 382]]), "MLP\n(k)"),
    (3, 2): (np.array([[34, 1703],[  0, 1737]]), "SGD\n(l)"),
    (4, 0): (np.array([[1314, 423],[1480, 257]]), "RF\n(m)"),
    (4, 1): (np.array([[873, 864],[1021, 716]]), "VC\n(n)"),
}

def plot_cm(ax, cm, title):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        cbar=False, square=True,
        linewidths=0.6, linecolor="white",
        annot_kws={"fontsize": ANN_FONTSIZE},
        ax=ax
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=8)

    ax.set_xticklabels(["BENIGN", "MALICIOUS"], fontsize=TICK_FONTSIZE, rotation=0, ha="center", color="purple")
    ax.set_yticklabels(["BENIGN", "MALICIOUS"], fontsize=TICK_FONTSIZE, rotation=90, va="center", color="purple")
    ax.tick_params(axis='x', pad=2)

    ax.set_xlabel("")
    ax.set_ylabel("")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

fig, axes = plt.subplots(5, 3, figsize=FIGSIZE)

# Apaga el hueco que te sobra
axes[4, 2].axis("off")

for (i, j), (cm, title) in cms.items():
    plot_cm(axes[i, j], cm, title)

# Labels globales (solo una vez)
fig.supxlabel("Predicted Values", fontsize=18)
fig.supylabel("True Values", fontsize=18)


# Ajuste fino de espaciado (mejor que tight_layout aquí)
fig.subplots_adjust(hspace=0.75, wspace=0.55, top=0.94, bottom=0.06)

plt.savefig("figura2.png", dpi=SAVE_DPI, bbox_inches="tight")
plt.close()
