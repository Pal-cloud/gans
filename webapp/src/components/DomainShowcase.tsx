"use client";

import { useState } from "react";

type Domain = {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  architectures: { name: string; desc: string }[];
  challenges: string[];
  color: string;
  svgContent: React.ReactNode;
};

const domains: Domain[] = [
  {
    id: "images",
    title: "Imagenes",
    subtitle: "El dominio con mayor impacto",
    description:
      "Las imagenes son el dominio donde las GANs han logrado los resultados mas espectaculares. Al ser datos continuos con estructura espacial, se adaptan perfectamente a las redes convolucionales. Hoy existen caras sinteticas indistinguibles de fotografias reales.",
    architectures: [
      { name: "DCGAN", desc: "Arquitectura base con convoluciones. Genera imagenes coherentes a baja resolucion." },
      { name: "StyleGAN 2/3", desc: "Control fino del estilo a distintas escalas. Imagenes fotorrealistas de alta resolucion." },
      { name: "Pix2Pix", desc: "Transformacion condicionada: boceto a fotografia, mapa a satelite." },
      { name: "CycleGAN", desc: "Transformacion sin pares de entrenamiento: caballos a cebras, verano a invierno." },
      { name: "SRGAN", desc: "Super-resolucion: aumentar la resolucion con detalles sinteticos plausibles." },
    ],
    challenges: [
      "Inestabilidad durante el entrenamiento",
      "Colapso de modo: el generador produce variedad reducida",
      "Evaluacion objetiva de la calidad visual",
    ],
    color: "violet",
    svgContent: null,
  },
  {
    id: "text",
    title: "Texto",
    subtitle: "El reto de la discrecion",
    description:
      "El texto es discreto: las palabras son unidades indivisibles y no existe nada 'entre' dos palabras. Esto impide que el gradiente fluya directamente, lo que hace que las GANs para texto requieran tecnicas especiales como aprendizaje por refuerzo o trabajar en el espacio de embeddings.",
    architectures: [
      { name: "SeqGAN", desc: "Usa Monte Carlo Tree Search para estimar el gradiente sobre secuencias discretas." },
      { name: "TextGAN", desc: "Trabaja en el espacio continuo de embeddings para permitir el flujo de gradiente." },
      { name: "MaskGAN", desc: "Rellena partes enmascaradas de un texto de forma coherente con el contexto." },
      { name: "Transformer-GAN", desc: "Combina la arquitectura Transformer con el esquema adversarial." },
    ],
    challenges: [
      "La no diferenciabilidad de la seleccion de palabras discretas",
      "Dificultad para mantener coherencia a largo plazo",
      "Los LLMs modernos (GPT) han superado a las GANs en la mayoria de tareas de texto",
    ],
    color: "blue",
    svgContent: null,
  },
  {
    id: "audio",
    title: "Audio",
    subtitle: "Onda y espectrograma",
    description:
      "El audio puede tratarse como datos temporales (forma de onda) o transformarse en una imagen 2D (espectrograma). La segunda aproximacion permite reutilizar directamente las tecnicas de imagenes. El oido humano es muy sensible a discontinuidades, lo que hace la evaluacion mas exigente.",
    architectures: [
      { name: "WaveGAN", desc: "Opera directamente sobre la forma de onda con convoluciones 1D de kernel grande." },
      { name: "MelGAN", desc: "Genera espectrogramas Mel tratando el audio como una imagen 2D." },
      { name: "HiFi-GAN", desc: "Vocoder de alta fidelidad con multiples discriminadores a distintas escalas temporales." },
      { name: "MuseGAN", desc: "Generacion musical polifonica: coordina multiples instrumentos en el tiempo." },
    ],
    challenges: [
      "La coherencia temporal: el oido detecta discontinuidades que el ojo ignora",
      "Secuencias muy largas: 44.100 muestras por segundo en audio de calidad CD",
      "Artefactos sonoros como clicks y distorsion son muy perceptibles",
    ],
    color: "cyan",
    svgContent: null,
  },
];

const colorStyles: Record<string, { tab: string; active: string; badge: string; border: string }> = {
  violet: {
    tab: "border-violet-500 text-violet-400 bg-violet-500/10",
    active: "bg-violet-500/10 border-violet-500/30",
    badge: "bg-violet-500/20 text-violet-300",
    border: "border-violet-500/20",
  },
  blue: {
    tab: "border-blue-500 text-blue-400 bg-blue-500/10",
    active: "bg-blue-500/10 border-blue-500/30",
    badge: "bg-blue-500/20 text-blue-300",
    border: "border-blue-500/20",
  },
  cyan: {
    tab: "border-cyan-500 text-cyan-400 bg-cyan-500/10",
    active: "bg-cyan-500/10 border-cyan-500/30",
    badge: "bg-cyan-500/20 text-cyan-300",
    border: "border-cyan-500/20",
  },
};

export default function DomainShowcase() {
  const [activeDomain, setActiveDomain] = useState<string>("images");
  const domain = domains.find((d) => d.id === activeDomain)!;
  const styles = colorStyles[domain.color];

  return (
    <section className="py-24 px-6 bg-slate-900/30">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Dominios de aplicacion
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Las GANs se han aplicado a tres grandes dominios con estrategias y
            arquitecturas especificas para cada uno.
          </p>
        </div>

        {/* Selector de dominio */}
        <div className="flex gap-3 mb-10">
          {domains.map((d) => (
            <button
              key={d.id}
              onClick={() => setActiveDomain(d.id)}
              className={`flex-1 py-3 px-4 rounded-xl border text-sm font-semibold transition-all duration-200 ${
                activeDomain === d.id
                  ? colorStyles[d.color].tab
                  : "border-slate-800 text-slate-500 hover:border-slate-700 hover:text-slate-400"
              }`}
            >
              {d.title}
            </button>
          ))}
        </div>

        {/* Contenido del dominio */}
        <div className={`rounded-2xl border p-8 ${styles.active}`}>
          <div className="mb-6">
            <p className="text-xs uppercase tracking-widest text-slate-500 mb-1">
              {domain.subtitle}
            </p>
            <h3 className="text-2xl font-bold text-white mb-3">
              GANs para {domain.title}
            </h3>
            <p className="text-slate-300 leading-relaxed max-w-3xl">
              {domain.description}
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mt-8">
            {/* Arquitecturas */}
            <div>
              <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
                Arquitecturas principales
              </h4>
              <div className="space-y-3">
                {domain.architectures.map((arch) => (
                  <div
                    key={arch.name}
                    className="flex gap-3 p-3 rounded-xl bg-slate-900/50 border border-slate-800"
                  >
                    <span
                      className={`flex-shrink-0 px-2 py-0.5 rounded text-xs font-bold ${styles.badge}`}
                    >
                      {arch.name}
                    </span>
                    <p className="text-slate-400 text-sm leading-snug">
                      {arch.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Desafios */}
            <div>
              <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
                Desafios especificos
              </h4>
              <ul className="space-y-3">
                {domain.challenges.map((challenge, i) => (
                  <li
                    key={i}
                    className="flex gap-3 p-3 rounded-xl bg-slate-900/50 border border-slate-800"
                  >
                    <span className="w-5 h-5 rounded-full bg-slate-700 text-slate-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">
                      {i + 1}
                    </span>
                    <p className="text-slate-400 text-sm leading-snug">
                      {challenge}
                    </p>
                  </li>
                ))}
              </ul>

              <div className={`mt-6 p-4 rounded-xl border ${styles.border} bg-slate-900/30`}>
                <p className="text-xs uppercase tracking-widest text-slate-500 mb-2">
                  Recurso de referencia
                </p>
                <p className="text-sm text-slate-400">
                  {domain.id === "images" && "ThisPersonDoesNotExist.com — caras fotorrealistas generadas con StyleGAN2"}
                  {domain.id === "text" && "SeqGAN (Yu et al., 2017) — AAAI — primera GAN exitosa para texto secuencial"}
                  {domain.id === "audio" && "HiFi-GAN (Kong et al., 2020) — NeurIPS — vocoder de alta fidelidad para TTS"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
