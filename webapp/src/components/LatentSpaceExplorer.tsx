"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// Genera una "imagen" 8x8 de digito simulado a partir de parametros
function generateDigit(
  params: number[],
  canvas: HTMLCanvasElement
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const size = canvas.width;
  const cellSize = size / 8;

  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, size, size);

  // Simula un digito 8x8 usando los parametros latentes como "pesos"
  const p0 = (params[0] + 1) / 2; // normalizar a [0,1]
  const p1 = (params[1] + 1) / 2;
  const p2 = (params[2] + 1) / 2;
  const p3 = (params[3] + 1) / 2;

  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const x = col / 7;
      const y = row / 7;

      // Funcion que simula la distribucion de intensidad de un digito
      const intensity =
        Math.sin(x * Math.PI * (1 + p0 * 3)) *
        Math.cos(y * Math.PI * (1 + p1 * 2)) *
        0.5 +
        Math.abs(Math.sin((x + y) * Math.PI * (1 + p2 * 2))) *
        p3 * 0.5 +
        0.1;

      const clampedIntensity = Math.max(0, Math.min(1, intensity));
      const gray = Math.round(clampedIntensity * 230);

      ctx.fillStyle = `rgb(${gray},${gray},${Math.round(gray * 0.8)})`;
      ctx.fillRect(
        col * cellSize,
        row * cellSize,
        cellSize - 1,
        cellSize - 1
      );
    }
  }
}

export default function LatentSpaceExplorer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [params, setParams] = useState<number[]>([0, 0, 0, 0]);
  const [isAnimating, setIsAnimating] = useState<boolean>(false);
  const animRef = useRef<number | null>(null);
  const timeRef = useRef<number>(0);

  const paramLabels = [
    "Dimension 1 (forma general)",
    "Dimension 2 (grosor trazo)",
    "Dimension 3 (inclinacion)",
    "Dimension 4 (textura)",
  ];

  const draw = useCallback(() => {
    if (canvasRef.current) {
      generateDigit(params, canvasRef.current);
    }
  }, [params]);

  useEffect(() => {
    draw();
  }, [draw]);

  const startAnimation = () => {
    setIsAnimating(true);
    const animate = (timestamp: number) => {
      const t = timestamp / 1000;
      timeRef.current = t;
      setParams([
        Math.sin(t * 0.7),
        Math.cos(t * 0.5),
        Math.sin(t * 0.3 + 1),
        Math.cos(t * 0.9 + 2),
      ]);
      animRef.current = requestAnimationFrame(animate);
    };
    animRef.current = requestAnimationFrame(animate);
  };

  const stopAnimation = () => {
    setIsAnimating(false);
    if (animRef.current !== null) {
      cancelAnimationFrame(animRef.current);
      animRef.current = null;
    }
  };

  const randomize = () => {
    setParams([
      Math.random() * 2 - 1,
      Math.random() * 2 - 1,
      Math.random() * 2 - 1,
      Math.random() * 2 - 1,
    ]);
  };

  useEffect(() => {
    return () => {
      if (animRef.current !== null) cancelAnimationFrame(animRef.current);
    };
  }, []);

  return (
    <section
      id="espacio-latente"
      className="py-24 px-6 bg-slate-900/50 border-y border-slate-800"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Explorador del espacio latente
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Mueve los controles para cambiar los parametros del vector de
            ruido y observa como varia la muestra generada en tiempo real.
            Cada dimension controla un atributo diferente.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-10 items-start">
          {/* Canvas de visualizacion */}
          <div className="flex flex-col items-center gap-4">
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={256}
                height={256}
                className="rounded-xl border border-slate-700 shadow-2xl"
                style={{ imageRendering: "pixelated" }}
              />
              <div className="absolute top-3 left-3 bg-slate-900/80 rounded-md px-2 py-1 text-xs text-slate-400">
                Muestra generada
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={isAnimating ? stopAnimation : startAnimation}
                className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-colors duration-200 ${
                  isAnimating
                    ? "bg-red-600/20 border border-red-600 text-red-400 hover:bg-red-600/30"
                    : "bg-violet-600/20 border border-violet-600 text-violet-400 hover:bg-violet-600/30"
                }`}
              >
                {isAnimating ? "Detener animacion" : "Animar recorrido"}
              </button>
              <button
                onClick={randomize}
                disabled={isAnimating}
                className="px-5 py-2.5 rounded-lg text-sm font-medium border border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white disabled:opacity-30 transition-colors duration-200"
              >
                Aleatorio
              </button>
            </div>
          </div>

          {/* Controles */}
          <div className="space-y-6">
            <p className="text-xs uppercase tracking-widest text-slate-500 mb-2">
              Vector latente z = [{params.map((p) => p.toFixed(2)).join(", ")}]
            </p>

            {params.map((value, i) => (
              <div key={i}>
                <div className="flex justify-between mb-2">
                  <label className="text-sm text-slate-300">
                    {paramLabels[i]}
                  </label>
                  <span className="text-sm text-violet-400 font-mono">
                    {value.toFixed(3)}
                  </span>
                </div>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.01"
                  value={value}
                  disabled={isAnimating}
                  onChange={(e) => {
                    const newParams = [...params];
                    newParams[i] = parseFloat(e.target.value);
                    setParams(newParams);
                  }}
                  className="w-full h-2 rounded-full appearance-none bg-slate-700 accent-violet-500 disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
                />
              </div>
            ))}

            <div className="mt-6 p-4 rounded-xl bg-slate-900 border border-slate-800">
              <p className="text-xs text-slate-500 leading-relaxed">
                En una GAN real, el espacio latente tiene 100 o mas dimensiones.
                Cada una de ellas controla un aspecto de la muestra generada:
                en imagenes de caras, algunas dimensiones controlan la edad,
                otras el color del cabello, otras la orientacion del rostro.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
