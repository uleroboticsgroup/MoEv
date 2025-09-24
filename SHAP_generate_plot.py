import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Cargar las dos imágenes
img1 = mpimg.imread("/home/alberto/Descargas/shap-mlp-no-sampling.png")
img2 = mpimg.imread("/home/alberto/Descargas/shap-mlp-sampling.png")

# Crear figura con dos subplots en fila
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # ajusta figsize según necesites

# Mostrar cada imagen en un eje
axes[0].imshow(img1)
axes[0].axis("off")
axes[1].imshow(img2)
axes[1].axis("off")

# Ajustar márgenes para reducir espacio en blanco
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.1)

# Añadir etiquetas debajo de cada imagen
fig.text(0.3, 0.02, "(a)", ha="center", fontsize=22)  # debajo de la 1ª
fig.text(0.8, 0.02, "(b)", ha="center", fontsize=22)  # debajo de la 2ª

plt.savefig("figura_combinada.png", dpi=300, bbox_inches="tight")