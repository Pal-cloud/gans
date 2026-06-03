"""
GAN Minima en PyTorch
---------------------
Este script implementa la GAN original de Goodfellow et al. (2014)
usando capas completamente conectadas (lineales) sobre el dataset MNIST.

Objetivo didactico: entender el ciclo de entrenamiento adversarial
sin la complejidad adicional de las capas convolucionales.

Uso:
    python gan_simple.py

Las imagenes generadas se guardan en la carpeta ./output_simple/
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# ------------------------------------------------------------------ #
#  Configuracion general
# ------------------------------------------------------------------ #
LATENT_DIM   = 100      # Dimension del espacio latente (ruido de entrada)
IMG_SIZE     = 28 * 28  # MNIST son imagenes de 28x28 pixeles aplanadas
BATCH_SIZE   = 128
NUM_EPOCHS   = 50
LR           = 0.0002
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR   = "./output_simple"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Entrenando en: {DEVICE}")


# ------------------------------------------------------------------ #
#  Arquitectura del Generador
# ------------------------------------------------------------------ #
class Generator(nn.Module):
    """
    El Generador transforma un vector de ruido z en una imagen sintetica.
    Entrada:  vector de dimension LATENT_DIM con valores aleatorios
    Salida:   vector de dimension IMG_SIZE (imagen aplanada) en rango [-1, 1]
    """
    def __init__(self, latent_dim: int = LATENT_DIM, img_size: int = IMG_SIZE):
        super().__init__()
        self.model = nn.Sequential(
            # Capa 1: proyecta el ruido a un espacio intermedio
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, momentum=0.8),

            # Capa 2: aumenta la capacidad representacional
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, momentum=0.8),

            # Capa 3: expande a la dimension de imagen
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),

            # Capa de salida: Tanh normaliza entre -1 y 1
            nn.Linear(1024, img_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


# ------------------------------------------------------------------ #
#  Arquitectura del Discriminador
# ------------------------------------------------------------------ #
class Discriminator(nn.Module):
    """
    El Discriminador evalua si una imagen es real (1) o generada (0).
    Entrada:  vector de dimension IMG_SIZE (imagen aplanada)
    Salida:   escalar en rango [0, 1] (probabilidad de ser real)
    """
    def __init__(self, img_size: int = IMG_SIZE):
        super().__init__()
        self.model = nn.Sequential(
            # Capa 1
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Capa 2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Capa de salida: Sigmoid produce una probabilidad
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.model(img)


# ------------------------------------------------------------------ #
#  Carga del dataset MNIST
# ------------------------------------------------------------------ #
transform = transforms.Compose([
    transforms.ToTensor(),
    # Normaliza de [0,1] a [-1,1] para que coincida con Tanh del Generador
    transforms.Normalize([0.5], [0.5]),
])

dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# ------------------------------------------------------------------ #
#  Inicializacion de modelos y optimizadores
# ------------------------------------------------------------------ #
generator     = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# Adam es el optimizador estandar para GANs
optimizer_G = torch.optim.Adam(generator.parameters(),     lr=LR, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# Binary Cross Entropy: mide que tan bien distingue el Discriminador
criterion = nn.BCELoss()

# Ruido fijo para visualizar la evolucion del Generador a lo largo del tiempo
fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)


# ------------------------------------------------------------------ #
#  Bucle de entrenamiento
# ------------------------------------------------------------------ #
history = {"d_loss": [], "g_loss": []}

for epoch in range(NUM_EPOCHS):
    d_loss_total = 0.0
    g_loss_total = 0.0

    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs  = real_imgs.view(batch_size, -1).to(DEVICE)  # aplanar imagen

        # Etiquetas: 1 = real, 0 = falso
        real_labels = torch.ones(batch_size,  1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # ---- Paso 1: Entrenar el Discriminador ---- #
        # Generar imagenes falsas (detach para no propagar al Generador)
        z         = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z).detach()

        # Perdida sobre muestras reales + perdida sobre muestras falsas
        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs), fake_labels)
        d_loss    = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---- Paso 2: Entrenar el Generador ---- #
        # El Generador quiere que el Discriminador clasifique sus imagenes como reales
        z         = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z)
        g_loss    = criterion(discriminator(fake_imgs), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

    # Promedios de la epoca
    avg_d = d_loss_total / len(dataloader)
    avg_g = g_loss_total / len(dataloader)
    history["d_loss"].append(avg_d)
    history["g_loss"].append(avg_g)

    print(f"Epoca [{epoch+1:3d}/{NUM_EPOCHS}]  "
          f"Perdida D: {avg_d:.4f}  |  Perdida G: {avg_g:.4f}")

    # Guardar muestras generadas cada 10 epocas
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            samples = generator(fixed_noise).view(-1, 1, 28, 28)
            # Desnormalizar de [-1,1] a [0,1] para mostrar correctamente
            samples = (samples + 1) / 2
        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=False)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"Imagenes generadas - Epoca {epoch+1}")
        plt.savefig(f"{OUTPUT_DIR}/epoch_{epoch+1:03d}.png", bbox_inches="tight")
        plt.close()

# ------------------------------------------------------------------ #
#  Guardar la curva de perdidas
# ------------------------------------------------------------------ #
plt.figure(figsize=(10, 4))
plt.plot(history["d_loss"], label="Perdida Discriminador")
plt.plot(history["g_loss"], label="Perdida Generador")
plt.xlabel("Epoca")
plt.ylabel("Perdida")
plt.title("Evolucion de las perdidas durante el entrenamiento")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", bbox_inches="tight")
plt.close()

print(f"\nEntrenamiento completado. Resultados guardados en {OUTPUT_DIR}/")
