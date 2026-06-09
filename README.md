# Redes Generativas Adversarias (GANs)
## Píldora formativa

---

## Tabla de contenidos

1. [Qué es una GAN](#1-qué-es-una-gan)
2. [Cómo funcionan las GANs por dentro](#2-cómo-funcionan-las-gans-por-dentro)
3. [Aplicaciones reales](#3-aplicaciones-reales)
4. [Archivos Python: descripción detallada](#4-archivos-python-descripción-detallada)
5. [Instalación de dependencias](#5-instalación-de-dependencias)
6. [CPU o GPU: cuál necesito](#6-cpu-o-gpu-cuál-necesito)
7. [Uso en imágenes](#7-uso-en-imágenes)
8. [Uso en textos](#8-uso-en-textos)
9. [Uso en audios](#9-uso-en-audios)
10. [Aplicación web interactiva](#10-aplicación-web-interactiva)
11. [Estructura del proyecto](#11-estructura-del-proyecto)
12. [Referencias](#12-referencias)

---

## 1. Qué es una GAN

Imagina que tienes dos personas trabajando juntas en un juego de engaño continuo. Una de ellas es un falsificador de billetes: su objetivo es fabricar billetes tan convincentes que parezcan reales. La otra persona es un detective: su trabajo es distinguir si un billete es auténtico o falsificado.

Al principio, el falsificador es muy malo en su trabajo y el detective lo detecta fácilmente. Pero cada vez que el detective rechaza un billete, el falsificador aprende de ese error y mejora su técnica. Del mismo modo, cada vez que el falsificador logra engañar al detective, el detective también aprende y se vuelve más exigente.

Este juego continúa durante miles o millones de rondas. Al final, el falsificador se vuelve tan bueno que sus billetes son indistinguibles de los auténticos, y el detective es tan preciso que puede detectar errores casi imperceptibles.

Este es exactamente el principio de funcionamiento de una Red Generativa Adversaria o GAN, propuesta por Ian Goodfellow y sus colegas en 2014.

### Los dos actores principales

**El Generador** es el falsificador. Parte de ruido aleatorio (números al azar) y, a través de un proceso de aprendizaje, aprende a transformar ese ruido en datos que parecen reales: imágenes de caras humanas que nunca han existido, fragmentos de música, textos coherentes, etcétera.

**El Discriminador** es el detective. Recibe dos tipos de muestras mezcladas: algunas provienen de datos reales (fotos reales de caras humanas, por ejemplo) y otras provienen del Generador. Su tarea es aprender a distinguir cuáles son reales y cuáles son falsas.

### Por qué es revolucionario

Antes de las GANs, crear contenido sintético de alta calidad era enormemente difícil. Los sistemas de inteligencia artificial podían clasificar o reconocer cosas, pero no eran buenos generando cosas nuevas. Las GANs cambiaron esto radicalmente al usar la competencia entre dos redes como mecanismo de aprendizaje.

El resultado es que hoy en día podemos generar:

- Fotografías realistas de personas que no existen
- Vídeos sintéticos con movimientos y expresiones naturales
- Música con estilos musicales específicos
- Arte digital con técnicas pictóricas concretas
- Voz sintética indistinguible de la voz humana
- Datos médicos sintéticos para entrenar otros modelos sin violar la privacidad

---

## 2. Cómo funcionan las GANs por dentro

### El ciclo de entrenamiento

El entrenamiento de una GAN es un proceso cíclico que se repite durante muchas iteraciones:

**Paso 1 - Entrenamiento del Discriminador:**
Se le presentan muestras reales del conjunto de datos y se le dice que son reales. Luego se le presentan muestras generadas por el Generador y se le dice que son falsas. El Discriminador ajusta sus parámetros internos para mejorar su capacidad de distinción.

**Paso 2 - Entrenamiento del Generador:**
El Generador produce muestras a partir de ruido aleatorio. Estas muestras pasan por el Discriminador, pero ahora el objetivo es que el Discriminador las clasifique como reales (aunque son falsas). El Generador ajusta sus parámetros para engañar al Discriminador.

**Paso 3 - Repetición:**
Este proceso se repite miles de veces. Con cada iteración, ambas redes mejoran en su tarea respectiva.

### El equilibrio de Nash

El objetivo teórico de una GAN es alcanzar lo que en teoría de juegos se llama el equilibrio de Nash: un punto en el que ningún jugador puede mejorar su situación cambiando unilateralmente su estrategia. En una GAN, esto ocurre cuando el Generador produce muestras tan buenas que el Discriminador ya no puede hacer nada mejor que adivinar al azar (50 % de probabilidad de acertar). En la práctica, este equilibrio es difícil de alcanzar y el entrenamiento puede ser inestable.

### Problemas comunes durante el entrenamiento

**Colapso de modo (Mode Collapse):** el Generador aprende a producir solo un tipo muy limitado de muestras porque estas engañan bien al Discriminador, ignorando la diversidad del conjunto de datos real.

**Desvanecimiento del gradiente:** si el Discriminador es demasiado bueno desde el principio, el Generador no recibe señal de aprendizaje suficiente y deja de mejorar.

**Inestabilidad:** el entrenamiento puede oscilar sin converger a un resultado estable.

---

## 3. Aplicaciones reales

### Generación de imágenes

Es el campo donde las GANs han tenido mayor impacto visible. El proyecto **https://ThisPersonDoesNotExist.com** muestra caras generadas por GANs que son indistinguibles de fotografías reales. Las arquitecturas como StyleGAN (desarrollada por NVIDIA) permiten controlar atributos específicos como la edad, el estilo del cabello o la expresión facial.

Otras aplicaciones incluyen:

- Transformación de estilo artístico (convertir una fotografía en una pintura al óleo)
- Aumento de resolución de imágenes antiguas o de baja calidad (superresolución)
- Transformación de día a noche o verano a invierno en fotografías
- Generación de imágenes médicas sintéticas (resonancias magnéticas, tomografías)

### Generación de texto

Aunque los modelos de lenguaje grandes (LLMs) como GPT han tomado protagonismo en la generación de texto, las GANs también se han aplicado en este dominio con arquitecturas como TextGAN y SeqGAN. El reto principal es que el texto es discreto (palabras o caracteres concretos) mientras que las GANs funcionan mejor con datos continuos como píxeles de imágenes.

Las aplicaciones incluyen generación de reseñas de productos, diálogos para videojuegos, y aumento de datos de texto para entrenar clasificadores.

### Generación de audio

Las GANs se han aplicado con éxito en síntesis de voz (WaveGAN, MelGAN), generación musical (MuseGAN), conversión de voz y mejora de calidad en grabaciones.

---

## 4. Archivos Python: descripción detallada

La carpeta `python/` contiene cuatro scripts con propósitos distintos. Todos comparten las mismas dependencias (véase la sección siguiente).

---

### `gan_simple.py` — GAN mínima con capas lineales

**Propósito:** implementación de referencia de la GAN original de Goodfellow et al. (2014). Usa exclusivamente capas completamente conectadas (lineales), sin convoluciones. Es el punto de partida más sencillo para entender el ciclo adversarial.

**Qué hace al ejecutarse:**
1. Descarga el dataset [MNIST](http://yann.lecun.com/exdb/mnist/) (60.000 imágenes de dígitos escritos a mano, 28 × 28 píxeles) la primera vez.
2. Entrena durante 50 épocas el Generador y el Discriminador de forma alternada (Una época = una vuelta completa al entrenamiento, de forma alterna, no entrenan a la vez, se turnan como en un juego).
3. Guarda una rejilla de imágenes generadas cada 5 épocas en `./output_simple/`.
4. Guarda la curva de pérdidas al finalizar en `./output_simple/loss_curve.png`.

**Arquitectura:**
- Generador: ruido (100) → capa lineal (256) → LeakyReLU → capa lineal (512) → LeakyReLU → capa lineal (784) → Tanh. Salida: imagen 28 × 28 aplanada.
- Discriminador: imagen (784) → capa lineal (512) → LeakyReLU → Dropout → capa lineal (256) → LeakyReLU → Dropout → capa lineal (1) → Sigmoid. Salida: probabilidad de ser real.

**Tiempo estimado de ejecución:** 15-30 minutos en CPU, 2-3 minutos en GPU.

```bash
python python/gan_simple.py
```

---

### `gan_images.py` — DCGAN convolucional sobre MNIST

**Propósito:** implementación de la arquitectura DCGAN (Deep Convolutional GAN) propuesta por Radford et al. (2015). Reemplaza las capas lineales por capas convolucionales, que explotan la estructura espacial de las imágenes. Es la versión más completa y la que produce mejores resultados visuales.

**Qué hace al ejecutarse:**
1. Descarga [MNIST](http://yann.lecun.com/exdb/mnist/) si no está ya disponible en `./data/`.
2. Inicializa los pesos siguiendo las recomendaciones del artículo original (distribución normal con media 0 y desviación estándar 0,02).
3. Entrena durante 50 épocas alternando Discriminador y Generador por cada lote.
4. Guarda imágenes de muestra cada 5 épocas en `./output_images/` (ficheros `dcgan_epoch_005.png`, etc.).
5. Guarda la curva de pérdidas en `./output_images/loss_curve.png`.
6. Guarda los pesos entrenados del Generador y del Discriminador en ficheros `.pth` para poder reutilizarlos sin volver a entrenar.

**Arquitectura:**
- Generador: ruido (100, 1, 1) → ConvTranspose2d (512, 4×4) → BatchNorm + ReLU → ConvTranspose2d (256, 7×7) → BatchNorm + ReLU → ConvTranspose2d (128, 14×14) → BatchNorm + ReLU → ConvTranspose2d (1, 28×28) → Tanh.
- Discriminador: imagen (1, 28×28) → Conv2d (128, 14×14) → LeakyReLU → Conv2d (256, 7×7) → BatchNorm + LeakyReLU → Conv2d (512, 3×3) → BatchNorm + LeakyReLU → Conv2d (1, 1×1) → Sigmoid.

**Tiempo estimado de ejecución:** 30-90 minutos en CPU, 5-10 minutos en GPU.

```bash
python python/gan_images.py
```

---

### `gan_text_conceptual.py` — Arquitectura conceptual para texto

**Propósito exclusivamente didáctico.** Este script no realiza un entrenamiento real ni descarga ningún corpus de texto. Su objetivo es mostrar cómo debe adaptarse la arquitectura GAN al dominio discreto del lenguaje, ilustrando el problema principal (la no diferenciabilidad de la selección de palabras) y dos estrategias para abordarlo.

**Qué hace al ejecutarse:**
1. Define un Generador de secuencias basado en LSTM que produce embeddings continuos.
2. Define un Discriminador que compara distribuciones de embeddings en lugar de trabajar con palabras discretas directamente.
3. Ejecuta un bucle de entrenamiento de prueba con datos sintéticos aleatorios (tensores de números aleatorios que simulan embeddings).
4. Imprime las pérdidas de cada iteración para mostrar que el ciclo adversarial funciona correctamente.
5. No genera texto real ni guarda ningún fichero de salida.

**Dependencias necesarias:** solo `torch` y `numpy` (incluidas en `requirements.txt`).

**Tiempo de ejecución:** menos de 30 segundos en cualquier máquina.

```bash
python python/gan_text_conceptual.py
```

---

### `gan_audio_conceptual.py` — Arquitectura conceptual para audio

**Propósito exclusivamente didáctico.** Al igual que el script anterior, no genera audio real ni utiliza datos de audio reales. Ilustra las dos aproximaciones principales para aplicar GANs al dominio del audio.

**Qué hace al ejecutarse:**
1. Define la arquitectura **WaveGAN**: Generador y Discriminador con convoluciones unidimensionales (1D) que operan directamente sobre la forma de onda.
2. Define la arquitectura **MelGAN**: Generador y Discriminador con convoluciones bidimensionales (2D) que operan sobre espectrogramas Mel, tratando el audio como si fuera una imagen.
3. Ejecuta una pasada hacia adelante con datos sintéticos aleatorios para verificar que las formas de los tensores son correctas.
4. Imprime las dimensiones de entrada y salida de cada red para facilitar la comprensión de la arquitectura.
5. No genera audio real ni guarda ningún fichero de salida.

**Dependencias necesarias:** solo `torch` y `numpy` (incluidas en `requirements.txt`).

**Tiempo de ejecución:** menos de 10 segundos en cualquier máquina.

```bash
python python/gan_audio_conceptual.py
```

---

## 5. Instalación de dependencias

Todos los scripts de la carpeta `python/` comparten el mismo fichero de dependencias.

### Requisitos

| Paquete | Versión mínima | Para qué se usa |
|---|---|---|
| `torch` | 2.0.0 | Motor de redes neuronales y autodiferenciación |
| `torchvision` | 0.15.0 | Descarga automática de MNIST y utilidades de imágenes |
| `numpy` | 1.24.0 | Operaciones numéricas auxiliares |
| `matplotlib` | 3.7.0 | Generación de gráficos y guardado de imágenes |
| `pillow` | 9.5.0 | Procesamiento de imágenes (requerido por torchvision) |
| `tqdm` | 4.65.0 | Barras de progreso durante el entrenamiento |

### Instalación

Desde la raíz del repositorio:

```bash
pip install -r python/requirements.txt
```

O de forma equivalente, instalando los paquetes uno a uno:

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

### Verificar la instalación

```bash
python -c "import torch; print(torch.__version__)"
```

Si el comando imprime el número de versión sin errores, la instalación es correcta.

---

## 6. CPU o GPU: cuál necesito

### Detección automática

Los scripts detectan automáticamente si hay una GPU disponible mediante:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

No es necesario modificar ningún parámetro: el script usará GPU si está disponible y CPU en caso contrario.

### Diferencias prácticas

| | CPU | GPU (NVIDIA con CUDA) |
|---|---|---|
| `gan_simple.py` (50 épocas) | 15-30 minutos | 2-3 minutos |
| `gan_images.py` (50 épocas) | 30-90 minutos | 5-10 minutos |
| Scripts conceptuales | menos de 1 minuto | menos de 1 minuto |

### Cuándo es suficiente la CPU

Para los fines didácticos de este repositorio, la CPU es completamente suficiente. Los scripts `gan_simple.py` y `gan_images.py` entrenan sobre [MNIST](http://yann.lecun.com/exdb/mnist/), un dataset pequeño y bien estudiado. Los tiempos de entrenamiento en CPU son asumibles en un ordenador moderno.

Si se quiere reducir el tiempo de espera sin GPU, basta con disminuir el número de épocas modificando la constante `NUM_EPOCHS` al inicio de cada script. Con 10 épocas se obtienen resultados ya visualmente comprensibles:

```python
NUM_EPOCHS = 10   # en lugar de 50
```

### Cuándo se necesita GPU

La GPU es necesaria o muy recomendable cuando se trabaja con:

- Datasets de imágenes de alta resolución (CelebA, LSUN, ImageNet)
- Arquitecturas profundas como StyleGAN2 o BigGAN
- Tiempos de entrenamiento de producción (cientos de épocas sobre millones de imágenes)

Para verificar si PyTorch detecta la GPU:

```python
python -c "import torch; print('GPU disponible:', torch.cuda.is_available())"
```

Si devuelve `False` en una máquina con GPU NVIDIA, es necesario instalar la versión de PyTorch con soporte CUDA desde https://pytorch.org/get-started/locally/.

---

## 7. Uso en imágenes

Las imágenes son el dominio donde las GANs han alcanzado resultados más espectaculares. El código en `python/gan_images.py` implementa una DCGAN que entrena sobre [MNIST](http://yann.lecun.com/exdb/mnist/) para generar dígitos escritos a mano.

### Por qué las imágenes se adaptan tan bien

Las imágenes son datos continuos (valores de píxeles) organizados en una estructura espacial regular. Esto las hace ideales para las GANs porque:

- El Discriminador puede aprender patrones visuales locales mediante capas convolucionales.
- Las métricas de similitud son intuitivas (coherencia de bordes, texturas).
- Los errores son visibles e interpretables por humanos durante el entrenamiento.

### DCGAN: la arquitectura estándar para imágenes

La DCGAN reemplaza las capas lineales por capas convolucionales y transpuestas. El Generador usa convoluciones transpuestas para aumentar progresivamente la resolución desde el vector de ruido hasta la imagen final. El Discriminador usa convoluciones normales para reducir la imagen a una probabilidad.

Reglas clave de diseño de una DCGAN:

- Usar Batch Normalization en todas las capas excepto la de entrada del Discriminador y la de salida del Generador. Modifica los datos en cada capa para que mantengan la misma escala y el entrenamiento no se vuelva loco (excepto al principio y al final).
- Usar LeakyReLU en el Discriminador y ReLU en el Generador.Funciones de activación. LeakyReLU deja pasar una pequeña parte de las señales negativas para que el Discriminador nunca deje de aprender.
- Usar Tanh como activación de salida del Generador.La función de salida del generador que escala los píxeles entre -1 y 1.

### Variantes importantes para imágenes

**Pix2Pix:** GAN condicionada que transforma imágenes de un dominio a otro. Por ejemplo, convierte bocetos en fotografías o mapas en imágenes satélite.

**CycleGAN:** permite transformar imágenes entre dos dominios sin necesitar pares de imágenes de entrenamiento. Convierte fotografías de caballos en cebras, o verano en invierno.

**StyleGAN2/3:** arquitectura de NVIDIA que permite controlar el estilo de imágenes a diferentes escalas de detalle. Produce los resultados más fotorrealistas conocidos.

**SRGAN:** especializada en superresolución: tomar imágenes de baja resolución y generar versiones de alta resolución con detalles sintéticos plausibles.

---

## 8. Uso en textos

La generación de texto con GANs presenta desafíos únicos frente a la generación de imágenes.

### El problema de la discreción

Las imágenes son continuas: un píxel puede tomar cualquier valor entre 0 y 255, y una pequeña perturbación produce una imagen ligeramente diferente. El texto es discreto: no existe nada entre la palabra «gato» y la palabra «perro». Este problema hace que el gradiente no pueda fluir directamente del Discriminador al Generador a través de las palabras generadas.

### Soluciones propuestas

**SeqGAN (2017):** trata la generación de texto como un proceso de toma de decisiones secuencial y usa técnicas de aprendizaje por refuerzo (Monte Carlo Tree Search) para estimar el gradiente a través de las palabras discretas.

**TextGAN:** trabaja en el espacio de representaciones continuas (embeddings) en lugar de trabajar directamente con palabras discretas, lo que permite el flujo de gradiente.

**BERT-GAN y variantes modernas:** combinan la arquitectura Transformer con el esquema adversarial para mejorar la coherencia a largo plazo del texto generado.

### Situación actual

Los modelos de lenguaje grandes (LLMs) como GPT han superado ampliamente a las GANs en la mayoría de tareas de generación de texto. Las GANs en texto tienen hoy un interés principalmente académico y en nichos específicos como la anonimización o el aumento de datos para clasificadores.

---

## 9. Uso en audios

El audio comparte con las imágenes la característica de ser datos continuos, pero presenta su propia complejidad temporal y frecuencial.

### Representaciones del audio para GANs

**Audio en bruto (forma de onda):** se trabaja directamente con la secuencia de muestras de amplitud. WaveGAN es el pionero de este enfoque. Requiere manejar secuencias muy largas (44.100 muestras por segundo para audio de calidad CD).

**Espectrograma:** se convierte el audio en una representación tiempo-frecuencia mediante la Transformada de Fourier de Tiempo Corto (STFT). Esto produce una imagen bidimensional que las GANs pueden procesar igual que cualquier imagen. MelGAN y HiFi-GAN utilizan espectrogramas Mel como representación intermedia.

### Arquitecturas especializadas

**WaveGAN:** adapta la DCGAN para operar sobre forma de onda unidimensional con convoluciones 1D.

**MelGAN:** genera espectrogramas Mel de alta calidad que luego se convierten a forma de onda. Es más eficiente que WaveGAN porque trabaja con una representación compacta del audio.

**HiFi-GAN:** arquitectura de síntesis de voz de alta fidelidad con múltiples discriminadores a diferentes escalas temporales.

**MuseGAN:** diseñada para generar música polifónica coordinando múltiples instrumentos.

---

## 10. Aplicación web interactiva

La aplicación web incluida en este repositorio permite explorar los conceptos de las GANs de forma visual e interactiva, sin necesidad de conocer programación ni instalar nada.

### Qué incluye la aplicación

- **Explicación paso a paso** del ciclo adversarial con la metáfora del falsificador y el detective.
- **Diagrama de arquitectura** animado con el flujo de datos completo.
- **Animación del entrenamiento** época a época con curva de pérdidas en tiempo real.
- **Explorador del espacio latente** con controles deslizantes para manipular el vector de ruido.
- **Sección de código** con el algoritmo completo anotado línea a línea.

### Tecnologías utilizadas

- HTML5, CSS3 y JavaScript sin dependencias externas ni frameworks.
- Canvas API para las visualizaciones interactivas.
- Servidor HTTP de Node.js en un único fichero (`webapp/server.js`), sin instalación adicional.

### Iniciar la aplicación

Desde la raíz del repositorio:

```bash
node webapp/server.js
```

La aplicación estará disponible en `http://localhost:3000`.

Para detener el servidor pulsar `Ctrl + C`.

---

## 11. Estructura del proyecto

```
gans/
├── README.md                          # Esta documentación
│
├── python/                            # Scripts Python ejecutables
│   ├── requirements.txt               # Dependencias comunes a todos los scripts
│   ├── gan_simple.py                  # GAN mínima con capas lineales (entrena sobre MNIST)
│   ├── gan_images.py                  # DCGAN convolucional (entrena sobre MNIST, mejor calidad)
│   ├── gan_text_conceptual.py         # Arquitectura conceptual para texto (solo didáctico)
│   └── gan_audio_conceptual.py        # Arquitectura conceptual para audio (solo didáctico)
│
└── webapp/                            # Aplicación web interactiva
    ├── server.js                      # Servidor HTTP estático (Node.js, sin dependencias)
    └── public/
        └── index.html                 # Toda la interfaz: HTML, CSS y JavaScript en un único fichero
```

---

## 12. Referencias

- MNIST dataset (http://yann.lecun.com/exdb/mnist/): base de datos de dígitos escritos a mano de LeCun et al. Dataset estándar utilizado para entrenar y evaluar los scripts `gan_simple.py` y `gan_images.py`.
- Curso-GANs: (https://github.com/nesmachnow/Curso-GANs) Repositorio de la Universidad de la República (Uruguay) con material teórico en español sobre fundamentos de redes neuronales generativas y resolución de problemas.
- GANs_coursera: (https://github.com/denisparra/GANs_coursera) Material y notas para el seguimiento del reconocido curso de especialización en GANs de DeepLearning.AI.
- gans-networks: (https://github.com/momartinm/gans-networks) Un proyecto introductorio que explica de forma sencilla qué son las Redes Generativas Antagónicas y su funcionamiento básico.
- This Person Does Not Exist (https://thispersondoesnotexist.com): demostración pública de StyleGAN.
