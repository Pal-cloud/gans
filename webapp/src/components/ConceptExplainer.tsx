"use client";

import { useState } from "react";

type Step = {
  id: number;
  title: string;
  description: string;
  metaphor: string;
  color: string;
};

const steps: Step[] = [
  {
    id: 1,
    title: "El Generador produce una muestra",
    description:
      "El Generador toma un vector de numeros aleatorios (ruido) y lo transforma en una muestra: una imagen, un fragmento de audio o un texto. Al principio del entrenamiento, estas muestras son completamente aleatorias e incoherentes.",
    metaphor:
      "El falsificador toma papel en blanco y, sin haber visto billetes reales, intenta fabricar uno de memoria. El resultado inicial es un garabato sin sentido.",
    color: "violet",
  },
  {
    id: 2,
    title: "El Discriminador evalua la muestra",
    description:
      "El Discriminador recibe la muestra generada junto con muestras reales del conjunto de datos. Analiza las caracteristicas y produce un numero entre 0 (falso) y 1 (real). Su objetivo es clasificar correctamente.",
    metaphor:
      "El detective examina el billete bajo la lupa. Detecta facilmente el error: el papel es incorrecto, la tinta no tiene relieve, el holograma no existe.",
    color: "blue",
  },
  {
    id: 3,
    title: "El Generador aprende del error",
    description:
      "La senal de error del Discriminador se propaga hacia atras hasta el Generador. Este ajusta sus parametros internos para producir muestras que sean mas dificiles de detectar como falsas en la siguiente iteracion.",
    metaphor:
      "El falsificador recibe la critica del detective: 'el papel es fino y no tiene marca de agua'. En el siguiente intento, usa papel mas grueso con textura.",
    color: "cyan",
  },
  {
    id: 4,
    title: "El Discriminador tambien mejora",
    description:
      "Ahora el Discriminador entrena de nuevo con muestras reales y las nuevas muestras mejoradas del Generador. Debe adaptarse para seguir distinguiendo correctamente, lo que lo hace mas exigente.",
    metaphor:
      "El detective, al ver billetes falsificados de mejor calidad, se forma con los expertos del banco central y aprende a detectar detalles mas sutiles.",
    color: "emerald",
  },
  {
    id: 5,
    title: "El ciclo se repite hasta el equilibrio",
    description:
      "El proceso se repite miles de veces. Al final, el Generador produce muestras tan buenas que el Discriminador ya no puede distinguirlas de las reales y acierta solo el 50% de las veces, es decir, acierta por azar.",
    metaphor:
      "Despues de anos de practica, los billetes del falsificador son perfectos. Ni el detector automatico ni el experto humano pueden distinguirlos. Se ha alcanzado el equilibrio.",
    color: "amber",
  },
];

const colorMap: Record<string, string> = {
  violet: "border-violet-500 bg-violet-500/10 text-violet-400",
  blue: "border-blue-500 bg-blue-500/10 text-blue-400",
  cyan: "border-cyan-500 bg-cyan-500/10 text-cyan-400",
  emerald: "border-emerald-500 bg-emerald-500/10 text-emerald-400",
  amber: "border-amber-500 bg-amber-500/10 text-amber-400",
};

const badgeMap: Record<string, string> = {
  violet: "bg-violet-500",
  blue: "bg-blue-500",
  cyan: "bg-cyan-500",
  emerald: "bg-emerald-500",
  amber: "bg-amber-500",
};

export default function ConceptExplainer() {
  const [activeStep, setActiveStep] = useState<number>(0);

  const step = steps[activeStep];

  return (
    <section
      id="concepto"
      className="py-24 px-6 bg-slate-900/50 border-y border-slate-800"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Como funciona una GAN
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Sin codigo y sin matematicas. Sigue el ciclo paso a paso usando la
            metafora del falsificador y el detective.
          </p>
        </div>

        {/* Navegacion por pasos */}
        <div className="flex gap-2 mb-10 overflow-x-auto pb-2">
          {steps.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setActiveStep(i)}
              className={`flex-shrink-0 flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium transition-all duration-200 ${
                activeStep === i
                  ? colorMap[s.color]
                  : "border-slate-700 text-slate-500 hover:border-slate-600 hover:text-slate-400"
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full flex items-center justify-center text-xs text-white font-bold ${
                  activeStep === i ? badgeMap[s.color] : "bg-slate-700"
                }`}
              >
                {s.id}
              </span>
              <span className="hidden md:inline">{s.title.split(" ")[2]}</span>
            </button>
          ))}
        </div>

        {/* Contenido del paso activo */}
        <div
          className={`rounded-2xl border p-8 transition-all duration-300 ${colorMap[step.color]}`}
          key={activeStep}
        >
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <span
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${badgeMap[step.color]}`}
                >
                  {step.id}
                </span>
                <h3 className="text-xl font-semibold text-white">
                  {step.title}
                </h3>
              </div>
              <p className="text-slate-300 leading-relaxed">
                {step.description}
              </p>
            </div>

            <div className="bg-slate-900/60 rounded-xl p-6 border border-slate-700">
              <p className="text-xs uppercase tracking-widest text-slate-500 mb-3">
                Metafora del falsificador
              </p>
              <p className="text-slate-300 leading-relaxed italic">
                {step.metaphor}
              </p>
            </div>
          </div>
        </div>

        {/* Botones de navegacion */}
        <div className="flex justify-between mt-6">
          <button
            onClick={() => setActiveStep((p) => Math.max(0, p - 1))}
            disabled={activeStep === 0}
            className="px-6 py-3 rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors duration-200"
          >
            Anterior
          </button>
          <span className="text-slate-600 text-sm self-center">
            {activeStep + 1} de {steps.length}
          </span>
          <button
            onClick={() =>
              setActiveStep((p) => Math.min(steps.length - 1, p + 1))
            }
            disabled={activeStep === steps.length - 1}
            className="px-6 py-3 rounded-lg border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors duration-200"
          >
            Siguiente
          </button>
        </div>
      </div>
    </section>
  );
}
