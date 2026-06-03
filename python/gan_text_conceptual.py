"""
GAN para texto - Arquitectura conceptual (SeqGAN simplificado)
--------------------------------------------------------------
Este script muestra la estructura de una GAN para generacion de texto.
No incluye entrenamiento completo porque requiere un corpus de texto
y muchas horas de computo. Su proposito es DIDACTICO: ilustrar como
se adapta la arquitectura GAN al dominio discreto del lenguaje.

El principal reto es que el texto es DISCRETO: no se puede calcular
el gradiente a traves de la operacion de seleccionar una palabra
(argmax o sampling), que es no diferenciable.

Soluciones implementadas aqui:
  - Trabajar en el espacio continuo de embeddings (no con palabras directamente)
  - El Discriminador compara distribuciones de embeddings, no palabras

Dependencias:
    pip install torch numpy

Uso:
    python gan_text_conceptual.py
"""

import torch
import torch.nn as nn
import numpy as np

# ------------------------------------------------------------------ #
#  Configuracion
# ------------------------------------------------------------------ #
VOCAB_SIZE   = 5000     # Tamano del vocabulario
EMBED_DIM    = 128      # Dimension de los embeddings de palabras
SEQ_LEN      = 20       # Longitud de la secuencia de texto
LATENT_DIM   = 64       # Dimension del espacio latente
HIDDEN_DIM   = 256      # Neuronas en capas ocultas
BATCH_SIZE   = 32
DEVICE       = "cpu"    # Cambiar a "cuda" si hay GPU disponible


# ------------------------------------------------------------------ #
#  Generador de texto basado en LSTM
# ------------------------------------------------------------------ #
class TextGenerator(nn.Module):
    """
    El Generador produce secuencias de embeddings a partir de ruido latente.

    Arquitectura:
      1. Un vector de ruido z inicializa el estado oculto del LSTM
      2. El LSTM genera una secuencia de vectores contextuales
      3. Una capa lineal proyecta cada vector al espacio de embeddings

    Nota: en lugar de producir palabras (indices discretos), el Generador
    produce vectores de embedding continuos. Esto permite que el gradiente
    fluya durante el entrenamiento.
    """
    def __init__(
        self,
        latent_dim: int  = LATENT_DIM,
        hidden_dim: int  = HIDDEN_DIM,
        embed_dim: int   = EMBED_DIM,
        seq_len: int     = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim

        # Proyectar el ruido al estado oculto inicial del LSTM
        self.noise_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.noise_to_cell   = nn.Linear(latent_dim, hidden_dim)

        # LSTM que genera la secuencia
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # Proyectar la salida del LSTM al espacio de embeddings
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh(),
        )

        # Embedding inicial (token de inicio de secuencia)
        self.start_token = nn.Parameter(torch.randn(1, embed_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: tensor de forma (batch_size, latent_dim)
        Retorna: tensor de forma (batch_size, seq_len, embed_dim)
        """
        batch_size = z.size(0)

        # Inicializar estado oculto del LSTM con el ruido latente
        h0 = self.noise_to_hidden(z).unsqueeze(0).repeat(2, 1, 1)
        c0 = self.noise_to_cell(z).unsqueeze(0).repeat(2, 1, 1)

        # Token de inicio de secuencia repetido para todo el batch
        current_input = self.start_token.expand(batch_size, -1).unsqueeze(1)

        outputs = []
        hidden  = (h0, c0)

        # Generacion autoregresiva paso a paso
        for _ in range(self.seq_len):
            out, hidden = self.lstm(current_input, hidden)
            embed       = self.output_layer(out)
            outputs.append(embed)
            current_input = embed  # La salida actual es la entrada del siguiente paso

        return torch.cat(outputs, dim=1)  # (batch, seq_len, embed_dim)


# ------------------------------------------------------------------ #
#  Discriminador de texto basado en Conv1D
# ------------------------------------------------------------------ #
class TextDiscriminator(nn.Module):
    """
    El Discriminador clasifica una secuencia de embeddings como real o generada.

    Arquitectura:
      - Convoluciones 1D con distintos tamanos de kernel (captura n-gramas)
      - Max pooling global para obtener un vector de tamano fijo
      - Capa lineal final que produce la probabilidad

    Las convoluciones 1D sobre embeddings son equivalentes a detectar
    patrones de n-gramas en la secuencia de texto.
    """
    def __init__(
        self,
        embed_dim: int  = EMBED_DIM,
        seq_len: int    = SEQ_LEN,
    ):
        super().__init__()

        # Tres conjuntos de filtros con diferentes receptive fields
        # kernel=2: detecta bigramas  | kernel=3: trigramas | kernel=4: 4-gramas
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(embed_dim, 128, kernel_size=4, padding=0)

        self.activation = nn.LeakyReLU(0.2)
        self.dropout     = nn.Dropout(0.3)

        # Despues del max pooling global cada conv aporta 128 caracteristicas
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor de forma (batch_size, seq_len, embed_dim)
        Retorna: tensor de forma (batch_size, 1)
        """
        # Conv1D espera (batch, channels, length) -> transponer
        x = x.permute(0, 2, 1)

        # Aplicar convoluciones y max pooling global
        f2 = self.activation(self.conv2(x)).max(dim=-1)[0]
        f3 = self.activation(self.conv3(x)).max(dim=-1)[0]
        f4 = self.activation(self.conv4(x)).max(dim=-1)[0]

        # Concatenar caracteristicas de distintas escalas
        features = torch.cat([f2, f3, f4], dim=-1)
        features = self.dropout(features)

        return self.classifier(features)


# ------------------------------------------------------------------ #
#  Demostracion de la arquitectura
# ------------------------------------------------------------------ #
def demo_arquitectura():
    """Muestra las dimensiones de los tensores en cada etapa."""
    print("=" * 60)
    print("ARQUITECTURA GAN PARA TEXTO - DEMOSTRACION")
    print("=" * 60)

    generator     = TextGenerator().to(DEVICE)
    discriminator = TextDiscriminator().to(DEVICE)

    # Contar parametros
    params_g = sum(p.numel() for p in generator.parameters())
    params_d = sum(p.numel() for p in discriminator.parameters())

    print(f"\nParametros del Generador:     {params_g:,}")
    print(f"Parametros del Discriminador: {params_d:,}")

    # Paso forward de ejemplo
    z           = torch.randn(BATCH_SIZE, LATENT_DIM).to(DEVICE)
    fake_seqs   = generator(z)

    print(f"\nRuido de entrada (z):         {z.shape}")
    print(f"Secuencias generadas:          {fake_seqs.shape}")
    print(f"  -> batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, embed_dim={EMBED_DIM}")

    prob        = discriminator(fake_seqs)
    print(f"\nProbabilidad (Discriminador): {prob.shape}")
    print(f"  -> valor medio: {prob.mean().item():.4f} (deberia tender a 0.5 al converger)")

    print("\nNota sobre la generacion de palabras reales:")
    print("  Para convertir embeddings a palabras, se usa la palabra mas cercana")
    print("  en el vocabulario mediante distancia coseno o producto escalar.")
    print("  Ejemplo (pseudocodigo):")
    print("    embedding_output = generator(z)  # (batch, seq_len, embed_dim)")
    print("    word_ids = (embedding_output @ vocab_matrix.T).argmax(dim=-1)")
    print("    words = [vocabulary[id] for id in word_ids[0]]")

    print("\nNota sobre SeqGAN (alternativa):")
    print("  SeqGAN genera palabras discretas y usa Monte Carlo Tree Search")
    print("  para estimar el gradiente. El Discriminador evalua la secuencia")
    print("  completa o parcial, y la senal de recompensa guia al Generador.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_arquitectura()
