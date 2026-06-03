"""
GAN para audio - Arquitectura conceptual (WaveGAN / MelGAN simplificados)
--------------------------------------------------------------------------
Este script muestra como se adapta la arquitectura GAN al dominio del audio.
Su proposito es DIDACTICO: ilustrar las dos aproximaciones principales.

APROXIMACION 1 - WaveGAN: trabaja directamente con la forma de onda (audio en bruto)
APROXIMACION 2 - MelGAN:  trabaja con espectrogramas Mel (representacion 2D del audio)

La segunda aproximacion trata el audio como si fuera una imagen,
lo que permite reutilizar las tecnicas de DCGAN ya conocidas.

Dependencias:
    pip install torch numpy

Uso:
    python gan_audio_conceptual.py
"""

import torch
import torch.nn as nn
import numpy as np

# ------------------------------------------------------------------ #
#  Configuracion
# ------------------------------------------------------------------ #
LATENT_DIM      = 100
SAMPLE_RATE     = 16000   # Hz
AUDIO_LEN       = 16384   # Muestras (~1 segundo a 16kHz)
MEL_BINS        = 80      # Numero de bandas Mel
MEL_FRAMES      = 128     # Numero de frames temporales del espectrograma
BATCH_SIZE      = 16
DEVICE          = "cpu"


# ================================================================== #
#  APROXIMACION 1: WaveGAN (dominio temporal)
# ================================================================== #

class WaveGenerator(nn.Module):
    """
    Genera audio en bruto (forma de onda) usando convoluciones transpuestas 1D.

    El Generador parte de un vector de ruido y lo expande progresivamente
    en la dimension temporal hasta alcanzar la longitud de audio deseada.

    Analogia con DCGAN:
      - DCGAN usa Conv2D transpuesta para aumentar resolucion de imagen
      - WaveGAN usa Conv1D transpuesta para aumentar longitud de audio
    """
    def __init__(self, latent_dim: int = LATENT_DIM, audio_len: int = AUDIO_LEN):
        super().__init__()

        # Calculo de la secuencia de upsampling:
        # Empezamos con audio_len / (stride^num_bloques) muestras
        # Con 4 bloques de stride=4: 16384 / 256 = 64 muestras iniciales

        self.main = nn.Sequential(
            # Bloque 1: proyeccion del ruido
            # in: (latent_dim, 1) -> out: (512, 64)
            nn.ConvTranspose1d(latent_dim, 512, kernel_size=25, stride=1, padding=12, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Bloque 2: x4 en longitud
            # in: (512, 64) -> out: (256, 256)
            nn.ConvTranspose1d(512, 256, kernel_size=25, stride=4, padding=11, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Bloque 3: x4
            # in: (256, 256) -> out: (128, 1024)
            nn.ConvTranspose1d(256, 128, kernel_size=25, stride=4, padding=11, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Bloque 4: x4
            # in: (128, 1024) -> out: (64, 4096)
            nn.ConvTranspose1d(128, 64, kernel_size=25, stride=4, padding=11, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Bloque 5: x4
            # in: (64, 4096) -> out: (1, 16384)
            nn.ConvTranspose1d(64, 1, kernel_size=25, stride=4, padding=11, bias=False),
            nn.Tanh(),  # Audio normalizado entre -1 y 1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim) -> audio: (batch, 1, audio_len)
        """
        return self.main(z.unsqueeze(-1))


class WaveDiscriminator(nn.Module):
    """
    Evalua si un fragmento de audio en bruto es real o generado.
    Usa convoluciones 1D con kernel grande para capturar patrones temporales.

    Diferencia con el Discriminador de imagen:
      - Los kernels son mas grandes (25 en lugar de 4) para cubrir
        suficientes ciclos de onda a la frecuencia de muestreo dada
    """
    def __init__(self, audio_len: int = AUDIO_LEN):
        super().__init__()
        self.main = nn.Sequential(
            # in: (1, 16384) -> out: (64, 4096)
            nn.Conv1d(1,   64,  kernel_size=25, stride=4, padding=11, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (64, 4096) -> out: (128, 1024)
            nn.Conv1d(64,  128, kernel_size=25, stride=4, padding=11, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (128, 1024) -> out: (256, 256)
            nn.Conv1d(128, 256, kernel_size=25, stride=4, padding=11, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (256, 256) -> out: (512, 64)
            nn.Conv1d(256, 512, kernel_size=25, stride=4, padding=11, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.main(x)
        return self.output(features)


# ================================================================== #
#  APROXIMACION 2: MelGAN (dominio espectrograma)
# ================================================================== #

class MelGenerator(nn.Module):
    """
    Genera espectrogramas Mel en lugar de audio en bruto.

    Un espectrograma Mel es una imagen 2D donde:
      - Eje X: tiempo (frames)
      - Eje Y: frecuencia (bandas Mel)
      - Valor: energia en esa banda y ese instante

    Al ser una imagen, podemos aplicar exactamente las mismas tecnicas
    que en la DCGAN de imagenes. El resultado (espectrograma) se convierte
    despues a audio mediante un vocoder (Griffin-Lim o HiFi-GAN).

    Esta aproximacion es mas eficiente que WaveGAN porque los espectrogramas
    son representaciones compactas: 80 x 128 pixeles en lugar de 16384 muestras.
    """
    def __init__(self, latent_dim: int = LATENT_DIM, mel_bins: int = MEL_BINS, mel_frames: int = MEL_FRAMES):
        super().__init__()
        self.main = nn.Sequential(
            # in: (latent_dim, 1, 1) -> out: (512, 5, 8)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=(5, 8), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # in: (512, 5, 8) -> out: (256, 10, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # in: (256, 10, 16) -> out: (128, 20, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # in: (128, 20, 32) -> out: (64, 40, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # in: (64, 40, 64) -> out: (1, 80, 128)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim) -> espectrograma: (batch, 1, mel_bins, mel_frames)
        """
        return self.main(z.unsqueeze(-1).unsqueeze(-1))


class MelDiscriminator(nn.Module):
    """
    Evalua si un espectrograma Mel es real o generado.
    Identico a un discriminador de imagen con Conv2D.
    """
    def __init__(self, mel_bins: int = MEL_BINS, mel_frames: int = MEL_FRAMES):
        super().__init__()
        self.main = nn.Sequential(
            # in: (1, 80, 128) -> out: (64, 40, 64)
            nn.Conv2d(1,   64,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (64, 40, 64) -> out: (128, 20, 32)
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (128, 20, 32) -> out: (256, 10, 16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (256, 10, 16) -> out: (512, 5, 8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # in: (512, 5, 8) -> out: (1, 1, 1)
            nn.Conv2d(512, 1, kernel_size=(5, 8), stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1)


# ------------------------------------------------------------------ #
#  Demostracion de las arquitecturas
# ------------------------------------------------------------------ #
def demo_arquitecturas():
    print("=" * 60)
    print("ARQUITECTURAS GAN PARA AUDIO - DEMOSTRACION")
    print("=" * 60)

    # --- WaveGAN ---
    print("\n--- APROXIMACION 1: WaveGAN (forma de onda) ---")
    wg = WaveGenerator().to(DEVICE)
    wd = WaveDiscriminator().to(DEVICE)

    z          = torch.randn(4, LATENT_DIM).to(DEVICE)
    fake_audio = wg(z)
    prob       = wd(fake_audio)

    print(f"Ruido entrada:         {z.shape}")
    print(f"Audio generado:        {fake_audio.shape}  ({fake_audio.shape[2]/SAMPLE_RATE:.2f}s a {SAMPLE_RATE}Hz)")
    print(f"Probabilidad D:        {prob.shape}")
    print(f"Parametros G:          {sum(p.numel() for p in wg.parameters()):,}")
    print(f"Parametros D:          {sum(p.numel() for p in wd.parameters()):,}")

    # --- MelGAN ---
    print("\n--- APROXIMACION 2: MelGAN (espectrograma Mel) ---")
    mg   = MelGenerator().to(DEVICE)
    md   = MelDiscriminator().to(DEVICE)

    z        = torch.randn(4, LATENT_DIM).to(DEVICE)
    fake_mel = mg(z)
    prob_mel = md(fake_mel)

    print(f"Ruido entrada:         {z.shape}")
    print(f"Espectrograma Mel:     {fake_mel.shape}  ({MEL_BINS} bandas x {MEL_FRAMES} frames)")
    print(f"Probabilidad D:        {prob_mel.shape}")
    print(f"Parametros G:          {sum(p.numel() for p in mg.parameters()):,}")
    print(f"Parametros D:          {sum(p.numel() for p in md.parameters()):,}")

    print("\n--- COMPARACION DE APROXIMACIONES ---")
    print(f"WaveGAN - muestras de audio por segundo: {SAMPLE_RATE:,}")
    print(f"MelGAN  - pixels del espectrograma:      {MEL_BINS * MEL_FRAMES:,}")
    print("MelGAN trabaja con ~10x menos datos, por eso es mas eficiente.")

    print("\n--- CONVERSION ESPECTROGRAMA -> AUDIO ---")
    print("Para convertir el espectrograma generado a audio se usa un vocoder:")
    print("  Opcion simple:   Griffin-Lim (algoritmo iterativo, sin red neuronal)")
    print("  Opcion calidad:  HiFi-GAN vocoder (red neuronal pre-entrenada)")
    print("  Codigo de ejemplo (requiere librosa):")
    print("    import librosa")
    print("    mel_db = fake_mel[0, 0].numpy()  # (80, 128)")
    print("    mel_power = librosa.db_to_power(mel_db)")
    print("    audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=16000)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_arquitecturas()
