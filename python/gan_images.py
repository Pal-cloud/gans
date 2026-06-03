"""
DCGAN (Deep Convolutional GAN) en PyTorch
------------------------------------------
Implementacion de la arquitectura DCGAN propuesta por Radford et al. (2015).
Usa capas convolucionales que explotan la estructura espacial de las imagenes.

Mejoras sobre la GAN simple (gan_simple.py):
  - Convoluciones transpuestas en el Generador para reconstruir la imagen
  - Convoluciones estandar en el Discriminador para extraer caracteristicas
  - Batch Normalization para estabilizar el entrenamiento
  - Sin capas completamente conectadas en el interior de las redes

Dataset: MNIST (digitos escritos a mano 28x28)
Imagenes generadas: se guardan en ./output_images/

Uso:
    python gan_images.py
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ------------------------------------------------------------------ #
#  Configuracion
# ------------------------------------------------------------------ #
LATENT_DIM  = 100
IMG_CHANNELS = 1       # MNIST es en escala de grises
IMG_SIZE    = 28
BATCH_SIZE  = 64
NUM_EPOCHS  = 50
LR          = 0.0002
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR  = "./output_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Entrenando en: {DEVICE}")


# ------------------------------------------------------------------ #
#  Funcion auxiliar: inicializacion de pesos
# ------------------------------------------------------------------ #
def init_weights(module: nn.Module):
    """
    Inicializacion recomendada por Radford et al.:
    - Pesos de conv y conv transpuesta: distribucion normal (media=0, std=0.02)
    - Batch Normalization: media=1, std=0.02, sesgo=0
    """
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(module.bias.data, 0)


# ------------------------------------------------------------------ #
#  Generador convolucional
# ------------------------------------------------------------------ #
class GeneratorDCGAN(nn.Module):
    """
    Arquitectura: vector de ruido -> serie de convoluciones transpuestas -> imagen

    Cada ConvTranspose2d duplica el tamano espacial (stride=2).
    La secuencia es: ruido (100,) -> (512,1,1) -> (256,4,4) -> (128,7,7) -> (1,28,28)
    """
    def __init__(self, latent_dim: int = LATENT_DIM, channels: int = IMG_CHANNELS):
        super().__init__()
        self.main = nn.Sequential(
            # Bloque 1: ruido -> mapa de caracteristicas inicial
            # in_channels=latent_dim, out_channels=512, kernel=4, stride=1, padding=0
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Forma: (512, 4, 4)

            # Bloque 2
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Forma: (256, 7, 7)

            # Bloque 3
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Forma: (128, 14, 14)

            # Bloque de salida: sin BatchNorm, activacion Tanh
            nn.ConvTranspose2d(128, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # Forma final: (1, 28, 28)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z tiene forma (batch, latent_dim) -> hay que darle forma (batch, latent_dim, 1, 1)
        return self.main(z.unsqueeze(-1).unsqueeze(-1))


# ------------------------------------------------------------------ #
#  Discriminador convolucional
# ------------------------------------------------------------------ #
class DiscriminatorDCGAN(nn.Module):
    """
    Arquitectura: imagen -> serie de convoluciones -> probabilidad escalar

    Cada Conv2d con stride=2 reduce el tamano espacial a la mitad.
    NO se usa MaxPooling, sino strides en las convoluciones.
    NO se usa BatchNorm en la primera capa (regla de la DCGAN).
    """
    def __init__(self, channels: int = IMG_CHANNELS):
        super().__init__()
        self.main = nn.Sequential(
            # Bloque 1: sin BatchNorm en la primera capa
            # in: (1, 28, 28) -> out: (128, 14, 14)
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 2
            # in: (128, 14, 14) -> out: (256, 7, 7)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Bloque 3
            # in: (256, 7, 7) -> out: (512, 3, 3)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa de salida: reduce a un escalar
            # in: (512, 3, 3) -> out: (1, 1, 1)
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.main(img).view(-1, 1)


# ------------------------------------------------------------------ #
#  Dataset y DataLoader
# ------------------------------------------------------------------ #
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset    = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# ------------------------------------------------------------------ #
#  Modelos, optimizadores y criterio
# ------------------------------------------------------------------ #
generator     = GeneratorDCGAN().to(DEVICE)
discriminator = DiscriminatorDCGAN().to(DEVICE)

# Aplicar inicializacion de pesos
generator.apply(init_weights)
discriminator.apply(init_weights)

optimizer_G = torch.optim.Adam(generator.parameters(),     lr=LR, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
criterion   = nn.BCELoss()

fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)


# ------------------------------------------------------------------ #
#  Entrenamiento
# ------------------------------------------------------------------ #
history = {"d_loss": [], "g_loss": []}

for epoch in range(NUM_EPOCHS):
    d_total = 0.0
    g_total = 0.0

    for real_imgs, _ in dataloader:
        batch_size  = real_imgs.size(0)
        real_imgs   = real_imgs.to(DEVICE)
        real_labels = torch.ones(batch_size,  1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # -- Entrenar Discriminador --
        z         = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z).detach()

        d_loss = (
            criterion(discriminator(real_imgs), real_labels) +
            criterion(discriminator(fake_imgs), fake_labels)
        ) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # -- Entrenar Generador --
        z         = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = generator(z)
        g_loss    = criterion(discriminator(fake_imgs), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        d_total += d_loss.item()
        g_total += g_loss.item()

    avg_d = d_total / len(dataloader)
    avg_g = g_total / len(dataloader)
    history["d_loss"].append(avg_d)
    history["g_loss"].append(avg_g)

    print(f"Epoca [{epoch+1:3d}/{NUM_EPOCHS}]  D: {avg_d:.4f}  G: {avg_g:.4f}")

    if (epoch + 1) % 5 == 0 or epoch == 0:
        with torch.no_grad():
            samples = generator(fixed_noise)
            samples = (samples + 1) / 2   # desnormalizar
        grid = torchvision.utils.make_grid(samples, nrow=8)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"DCGAN - Epoca {epoch+1}")
        plt.savefig(f"{OUTPUT_DIR}/dcgan_epoch_{epoch+1:03d}.png", bbox_inches="tight")
        plt.close()

# Guardar curva de perdidas
plt.figure(figsize=(10, 4))
plt.plot(history["d_loss"], label="Discriminador")
plt.plot(history["g_loss"], label="Generador")
plt.xlabel("Epoca")
plt.ylabel("Perdida")
plt.title("DCGAN - Curva de perdidas")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", bbox_inches="tight")
plt.close()

# Guardar el modelo entrenado
torch.save(generator.state_dict(),     f"{OUTPUT_DIR}/generator.pth")
torch.save(discriminator.state_dict(), f"{OUTPUT_DIR}/discriminator.pth")

print(f"\nEntrenamiento completado. Resultados en: {OUTPUT_DIR}/")
