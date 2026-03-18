import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Cargar las dos imágenes
img1 = mpimg.imread("plot-no-sampling.png")
img2 = mpimg.imread("plot-sampling-1000.png")
img3 = mpimg.imread("plot-sampling-500.png")

# Crear figura con grid: 2 filas, 2 columnas
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.05)

# Dos imágenes arriba (ocupan una columna cada una)
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img1)
ax1.axis("off")

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(img2)
ax2.axis("off")

# Tercera imagen abajo centrada (ocupa dos columnas)
ax3 = fig.add_subplot(gs[1, :])
ax3.imshow(img3)
ax3.axis("off")

# Añadir etiquetas
fig.text(0.35, 0.48, "(a)", ha="center", fontsize=22)
fig.text(0.74, 0.48, "(b)", ha="center", fontsize=22)
fig.text(0.54, 0.08, "(c)", ha="center", fontsize=22)

plt.savefig("figura_combinada.png", dpi=300, bbox_inches="tight")