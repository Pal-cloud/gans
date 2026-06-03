# Redes Generativas Adversarias (GANs)
## Pildora Formativa

---

## Tabla de Contenidos

1. [Introduccion conceptual sin codigo](#1-introduccion-conceptual-sin-codigo)
2. [Como funcionan las GANs por dentro](#2-como-funcionan-las-gans-por-dentro)
3. [Aplicaciones reales](#3-aplicaciones-reales)
4. [Implementacion en Python](#4-implementacion-en-python)
5. [Uso en imagenes](#5-uso-en-imagenes)
6. [Uso en textos](#6-uso-en-textos)
7. [Uso en audios](#7-uso-en-audios)
8. [Aplicacion web interactiva](#8-aplicacion-web-interactiva)
9. [Dockerizacion y despliegue](#9-dockerizacion-y-despliegue)
10. [Estructura del proyecto](#10-estructura-del-proyecto)
11. [Referencias](#11-referencias)

---

## 1. Introduccion conceptual sin codigo

### Que es una GAN

Imagina que tienes dos personas trabajando juntas en un juego de engano continuo. Una de ellas es un falsificador de billetes: su objetivo es fabricar billetes tan convincentes que parezcan reales. La otra persona es un detective: su trabajo es distinguir si un billete es autentico o falsificado.

Al principio, el falsificador es muy malo en su trabajo y el detective lo detecta facilmente. Pero cada vez que el detective rechaza un billete, el falsificador aprende de ese error y mejora su tecnica. Del mismo modo, cada vez que el falsificador logra enganar al detective, el detective tambien aprende y se vuelve mas exigente.

Este juego continua durante miles o millones de rondas. Al final, el falsificador se vuelve tan bueno que sus billetes son indistinguibles de los autenticos, y el detective es tan preciso que puede detectar errores casi imperceptibles.

Este es exactamente el principio de funcionamiento de una Red Generativa Adversaria o GAN, propuesta por Ian Goodfellow y sus colegas en 2014.

### Los dos actores principales

**El Generador** es el falsificador. Parte de ruido aleatorio (numeros al azar) y a traves de un proceso de aprendizaje, aprende a transformar ese ruido en datos que parecen reales: imagenes de caras humanas que nunca han existido, fragmentos de musica, textos coherentes, etcetera.

**El Discriminador** es el detective. Recibe dos tipos de muestras mezcladas: algunas provienen de datos reales (fotos reales de caras humanas, por ejemplo) y otras provienen del Generador. Su tarea es aprender a distinguir cuales son reales y cuales son falsas.

### Por que es revolucionario

Antes de las GANs, crear contenido sintetico de alta calidad era enormemente dificil. Los sistemas de inteligencia artificial podian clasificar o reconocer cosas, pero no eran buenos generando cosas nuevas. Las GANs cambiaron esto radicalmente al usar la competencia entre dos redes como mecanismo de aprendizaje.

El resultado es que hoy en dia podemos generar:

- Fotografias realistas de personas que no existen
- Videos sinteticos con movimientos y expresiones naturales
- Musica con estilos musicales especificos
- Arte digital con tecnicas pictoricas concretas
- Voz sintetica indistinguible de la voz humana
- Datos medicos sinteticos para entrenar otros modelos sin violar privacidad

### La metafora completa

Piensa en la GAN como un proceso de forja artistica. Un maestro artesano (el Discriminador) ha visto miles de obras autenticas y sabe perfectamente como luce una obra genuina. Un aprendiz (el Generador) intenta crear obras nuevas bajo la supervision critica del maestro. Cada critica del maestro hace al aprendiz mejorar. Cuando el aprendiz es tan bueno que el maestro ya no puede distinguir sus obras de las originales, se ha alcanzado el equilibrio.

---

## 2. Como funcionan las GANs por dentro

### El ciclo de entrenamiento

El entrenamiento de una GAN es un proceso ciclico que se repite durante muchas iteraciones:

**Paso 1 - Entrenamiento del Discriminador:**
Se le presentan muestras reales del conjunto de datos (por ejemplo, fotografias reales) y se le dice que son reales. Luego se le presentan muestras generadas por el Generador (fotografias falsas) y se le dice que son falsas. El Discriminador ajusta sus parametros internos para mejorar su capacidad de distincion.

**Paso 2 - Entrenamiento del Generador:**
El Generador produce muestras a partir de ruido aleatorio. Estas muestras pasan por el Discriminador, pero ahora el objetivo es que el Discriminador las clasifique como reales (aunque son falsas). El Generador ajusta sus parametros para enganar al Discriminador.

**Paso 3 - Repeticion:**
Este proceso se repite miles de veces. Con cada iteracion, ambas redes mejoran en su tarea respectiva.

### El equilibrio de Nash

El objetivo teorico de una GAN es alcanzar lo que en teoria de juegos se llama el equilibrio de Nash: un punto en el que ningun jugador puede mejorar su situacion cambiando unilateralmente su estrategia. En una GAN, esto ocurre cuando el Generador produce muestras tan buenas que el Discriminador ya no puede hacer nada mejor que adivinar al azar (50% de probabilidad de acertar). En la practica, este equilibrio es dificil de alcanzar y el entrenamiento puede ser inestable.

### Problemas comunes durante el entrenamiento

**Colapso de modo (Mode Collapse):** El Generador aprende a producir solo un tipo muy limitado de muestras porque estas enganan bien al Discriminador, ignorando la diversidad del conjunto de datos real.

**Desvanecimiento del gradiente:** Si el Discriminador es demasiado bueno desde el principio, el Generador no recibe senal de aprendizaje suficiente y deja de mejorar.

**Inestabilidad:** El entrenamiento puede oscilar sin converger a un resultado estable.

---

## 3. Aplicaciones reales

### Generacion de imagenes

Es el campo donde las GANs han tenido mayor impacto visible. El proyecto **ThisPersonDoesNotExist.com** muestra caras generadas por GANs que son indistinguibles de fotografias reales. Las arquitecturas como StyleGAN (desarrollada por NVIDIA) permiten controlar atributos especificos como la edad, el estilo del cabello o la expresion facial.

Otras aplicaciones incluyen:

- Transformacion de estilo artistico (convertir una fotografia en una pintura al oleo)
- Aumento de resolucion de imagenes antiguas o de baja calidad (super-resolucion)
- Transformacion de dia a noche o verano a invierno en fotografias
- Generacion de imagenes medicas sinteticas (resonancias magneticas, tomografias)

### Generacion de texto

Aunque los modelos de lenguaje grande (LLMs) como GPT han tomado protagonismo en la generacion de texto, las GANs tambien se han aplicado en este dominio con arquitecturas como TextGAN y SeqGAN. El reto principal es que el texto es discreto (palabras o caracteres concretos) mientras que las GANs funcionan mejor con datos continuos como pixeles de imagenes.

Las aplicaciones incluyen generacion de resenas de productos, dialogos para videojuegos, y augmentacion de datos de texto para entrenar clasificadores.

### Generacion de audio

Las GANs se han aplicado con exito en:

- **Sintesis de voz:** WaveGAN y MelGAN generan audio de voz realista
- **Generacion musical:** MuseGAN genera composiciones musicales en multiples instrumentos
- **Conversion de voz:** Cambiar las caracteristicas de una voz (genero, acento) manteniendo el contenido
- **Mejora de audio:** Eliminacion de ruido y mejora de calidad en grabaciones

---

## 4. Implementacion en Python

### Requisitos previos

```bash
pip install -r python/requirements.txt
```

### Estructura de una GAN minima en PyTorch

El codigo completo y ejecutable se encuentra en `python/gan_simple.py`. A continuacion se explican los bloques principales:

**Arquitectura del Generador:**

El Generador toma un vector de ruido de dimension fija (llamado espacio latente) y lo transforma en una muestra del tamano deseado. Cada capa aprende a darle estructura progresivamente mas compleja a ese ruido.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_size),
            nn.Tanh()  # normaliza la salida entre -1 y 1
        )

    def forward(self, z):
        return self.model(z)
```

**Arquitectura del Discriminador:**

El Discriminador recibe una muestra (real o generada) y devuelve una probabilidad de que sea real.

```python
class Discriminator(nn.Module):
    def __init__(self, img_size=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # probabilidad entre 0 y 1
        )

    def forward(self, img):
        return self.model(img)
```

**Bucle de entrenamiento simplificado:**

```python
for epoch in range(num_epochs):
    for real_imgs in dataloader:
        batch_size = real_imgs.size(0)

        # --- Entrenar Discriminador ---
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z).detach()

        real_loss = criterion(discriminator(real_imgs), ones)
        fake_loss = criterion(discriminator(fake_imgs), zeros)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # --- Entrenar Generador ---
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), ones)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
```

---

## 5. Uso en imagenes

Las imagenes son el dominio donde las GANs han alcanzado resultados mas espectaculares. El codigo en `python/gan_images.py` implementa una DCGAN (Deep Convolutional GAN) que entrena sobre el dataset MNIST para generar digitos escritos a mano.

### Por que las imagenes se adaptan tan bien

Las imagenes son datos continuos (valores de pixeles entre 0 y 255) organizados en una estructura espacial regular. Esto las hace ideales para las GANs porque:

- El Discriminador puede aprender patrones visuales locales mediante capas convolucionales
- Las metricas de similitud son intuitivas (distancia de pixeles, coherencia de bordes)
- Los errores son visibles e interpretables por humanos durante el entrenamiento

### DCGAN: la arquitectura estandar para imagenes

La DCGAN reemplaza las capas lineales por capas convolucionales y transpuestas. El Generador usa convoluciones transpuestas para aumentar progresivamente la resolucion desde el vector de ruido hasta la imagen final. El Discriminador usa convoluciones normales para reducir la imagen a una probabilidad.

Reglas clave de diseno de una DCGAN:

- Usar Batch Normalization en todas las capas excepto la de entrada del Discriminador y la de salida del Generador
- Usar LeakyReLU en el Discriminador y ReLU en el Generador
- Usar Tanh como activacion de salida del Generador

### Variantes importantes para imagenes

**Pix2Pix:** GAN condicionada que transforma imagenes de un dominio a otro. Por ejemplo, convierte bocetos en fotografias o mapas en imagenes satelite.

**CycleGAN:** Permite transformar imagenes entre dos dominios sin necesitar pares de imagenes de entrenamiento. Convierte fotografias de caballos en cebras, o verano en invierno.

**StyleGAN2/3:** Arquitectura de NVIDIA que permite controlar el estilo de imagenes a diferentes escalas de detalle. Produce los resultados mas fotorrealistas conocidos.

**SRGAN:** Especializada en super-resolucion: tomar imagenes de baja resolucion y generar versiones de alta resolucion con detalles sinteticos plausibles.

---

## 6. Uso en textos

La generacion de texto con GANs presenta desafios unicos frente a la generacion de imagenes.

### El problema de la discrecion

Las imagenes son continuas: un pixel puede tomar cualquier valor entre 0 y 255, y una pequena perturbacion produce una imagen ligeramente diferente. El texto es discreto: no existe nada entre la palabra "gato" y la palabra "perro". Este problema hace que el gradiente no pueda fluir directamente del Discriminador al Generador a traves de las palabras generadas.

### Soluciones propuestas

**SeqGAN (2017):** Trata la generacion de texto como un proceso de toma de decisiones secuencial y usa tecnicas de aprendizaje por refuerzo (Monte Carlo Tree Search) para estimar el gradiente a traves de las palabras discretas.

**TextGAN:** Trabaja en el espacio de representaciones continuas (embeddings) en lugar de trabajar directamente con palabras discretas, lo que permite el flujo de gradiente.

**BERT-GAN y variantes modernas:** Combinan la arquitectura Transformer con el esquema adversarial para mejorar la coherencia a largo plazo del texto generado.

### Aplicaciones practicas en texto

- Augmentacion de datos: generar ejemplos adicionales de texto para clases poco representadas en un clasificador
- Generacion de dialogos para videojuegos y asistentes virtuales
- Anonimizacion de textos manteniendo el estilo y estructura
- Generacion de resenas o descripciones de productos para pruebas de sistemas

---

## 7. Uso en audios

El audio comparte con las imagenes la caracteristica de ser datos continuos, pero presenta su propia complejidad temporal y frecuencial.

### Representaciones del audio para GANs

El audio puede introducirse en una GAN de dos formas principales:

**Audio en bruto (forma de onda):** Se trabaja directamente con la secuencia de muestras de amplitud. WaveGAN es el pionero de este enfoque. Requiere manejar secuencias muy largas (44.100 muestras por segundo para audio de calidad CD).

**Espectrograma:** Se convierte el audio en una representacion tiempo-frecuencia mediante la Transformada de Fourier de Tiempo Corto (STFT). Esto produce una imagen bidimensional que las GANs pueden procesar igual que cualquier imagen. MelGAN y HiFi-GAN utilizan espectrogramas Mel como representacion intermedia.

### Arquitecturas especializadas

**WaveGAN:** Adapta la DCGAN para operar sobre forma de onda unidimensional. Usa convoluciones 1D en lugar de 2D. Fue uno de los primeros modelos en generar audio musical y de voz de calidad aceptable directamente en el dominio temporal.

**MelGAN:** Genera espectrogramas Mel de alta calidad que luego se convierten a forma de onda. Es mas eficiente que WaveGAN porque trabaja con una representacion compacta del audio.

**HiFi-GAN:** Arquitectura de sintesis de voz de alta fidelidad que usa multiples discriminadores a diferentes escalas temporales para garantizar coherencia tanto a nivel de ciclos individuales de la onda como a nivel de frases completas.

**MuseGAN:** Disenada para generar musica polifonica (multiples instrumentos). Usa un modelo temporal para la progresion musical y un modelo de composicion para la coordinacion entre instrumentos.

### Desafios especificos del audio

- La coherencia temporal es critica: el oido humano detecta discontinuidades que el ojo podria ignorar
- Los artefactos sonoros (clicks, distorsion) son muy perceptibles
- La evaluacion objetiva de calidad es mas dificil que en imagenes

---

## 8. Aplicacion web interactiva

La aplicacion web incluida en este repositorio permite explorar los conceptos de las GANs de forma visual e interactiva, sin necesidad de conocer programacion.

### Que incluye la aplicacion

**Visualizacion del espacio latente:**
Una interfaz de sliders que permite al usuario manipular el vector de ruido de entrada (espacio latente) y ver en tiempo real como cambia la imagen generada. Esto ilustra como distintas dimensiones del espacio latente controlan distintos atributos visuales.

**Animacion del proceso de entrenamiento:**
Una visualizacion paso a paso que muestra como evoluciona el Generador a lo largo de las epocas de entrenamiento, desde imagenes completamente aleatorias hasta imagenes coherentes. Permite entender visualmente el concepto de convergencia.

**Comparacion real vs. generado:**
Un panel de comparacion lado a lado donde el usuario puede ver imagenes reales del conjunto de datos junto a imagenes generadas, apreciando el nivel de realismo alcanzado.

**Explicacion interactiva del flujo de datos:**
Un diagrama animado que muestra el recorrido de los datos a traves del Generador y el Discriminador, con indicacion del valor de perdida en cada paso.

### Tecnologias utilizadas

- Next.js 14 con App Router para la estructura de la aplicacion
- TypeScript para tipado estatico y mejor mantenibilidad
- Tailwind CSS para el diseno visual
- Canvas API para las visualizaciones interactivas
- TensorFlow.js para ejecutar modelos pre-entrenados directamente en el navegador

### Iniciar la aplicacion en local

```bash
cd webapp
npm install
npm run dev
```

La aplicacion estara disponible en `http://localhost:3000`.

---

## 9. Dockerizacion y despliegue

### Construir y ejecutar con Docker

```bash
# Construir la imagen
docker build -t gans-webapp ./webapp

# Ejecutar el contenedor
docker run -p 3000:3000 gans-webapp
```

### Despliegue con Docker Compose

```bash
docker-compose up --build
```

La aplicacion estara disponible en `http://localhost:3000`.

### Obtener una URL publica

**Opcion A: Vercel (recomendado para Next.js, gratuito)**

```bash
npm install -g vercel
cd webapp
vercel
```

Vercel detecta automaticamente Next.js y despliega con una URL publica tipo `https://gans-webapp-xxxx.vercel.app`.

**Opcion B: Railway con Docker**

1. Crear cuenta en railway.app
2. Conectar el repositorio de GitHub
3. Railway detecta el Dockerfile y despliega automaticamente
4. Se obtiene una URL publica tipo `https://gans-webapp.up.railway.app`

**Opcion C: Fly.io**

```bash
fly launch --dockerfile webapp/Dockerfile
fly deploy
```

### Variables de entorno para produccion

Crear un fichero `.env.production` en la carpeta `webapp`:

```env
NEXT_PUBLIC_APP_URL=https://tu-dominio.com
NODE_ENV=production
```

---

## 10. Estructura del proyecto

```
gans/
├── README.md                          # Esta documentacion
├── docker-compose.yml                 # Orquestacion de contenedores
│
├── python/                            # Ejemplos en Python
│   ├── requirements.txt               # Dependencias Python
│   ├── gan_simple.py                  # GAN minima con capas lineales
│   ├── gan_images.py                  # DCGAN para generacion de imagenes
│   ├── gan_text_conceptual.py         # Arquitectura conceptual para texto
│   └── gan_audio_conceptual.py        # Arquitectura conceptual para audio
│
└── webapp/                            # Aplicacion web interactiva
    ├── Dockerfile                     # Imagen Docker de produccion
    ├── package.json
    ├── tsconfig.json
    ├── tailwind.config.ts
    ├── next.config.ts
    └── src/
        ├── app/
        │   ├── layout.tsx             # Layout principal
        │   ├── page.tsx               # Pagina de inicio
        │   └── globals.css
        └── components/
            ├── Hero.tsx               # Introduccion visual
            ├── ConceptExplainer.tsx   # Explicacion interactiva con metafora
            ├── TrainingAnimation.tsx  # Animacion del entrenamiento
            ├── LatentSpaceExplorer.tsx# Explorador del espacio latente
            ├── DomainShowcase.tsx     # Galeria por dominio: imagen, texto, audio
            └── Architecture.tsx      # Diagrama de arquitectura GAN
```

---

## 11. Referencias

**Articulo original:**
Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., y Bengio, Y. (2014). Generative Adversarial Nets. *Advances in Neural Information Processing Systems*, 27.

**Arquitecturas de referencia:**
- Radford, A., Metz, L., y Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *ICLR 2016*.
- Karras, T., Laine, S., Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. *CVPR 2019*.
- Isola, P., Zhu, J.-Y., Zhou, T., y Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017*.
- Zhu, J.-Y., Park, T., Isola, P., y Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *ICCV 2017*.

**Audio:**
- Donahue, C., McAuley, J., y Puckette, M. (2018). Adversarial Audio Synthesis. *ICLR 2019*.
- Kumar, K. et al. (2019). MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis.
- Kong, J., Kim, J., y Bae, J. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.

**Texto:**
- Yu, L., Zhang, W., Wang, J., y Yu, Y. (2017). SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. *AAAI 2017*.

**Recursos adicionales:**
- GAN Zoo (github.com/hindupuravinash/the-gan-zoo): catalogo de variantes de GANs publicadas
- Papers With Code - GANs (paperswithcode.com): implementaciones y benchmarks actualizados
- This Person Does Not Exist (thispersondoesnotexist.com): demostracion publica de StyleGAN
