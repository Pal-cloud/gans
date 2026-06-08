"use client";

import { useState, useEffect, useRef } from "react";

type Frame = {
  epoch: number;
  dLoss: number;
  gLoss: number;
  quality: number; // 0-1, calidad visual simulada
};

/**
 * Generador pseudoaleatorio determinista basado en semilla (algoritmo mulberry32).
 * Produce siempre los mismos valores para la misma semilla, tanto en el servidor
 * (SSR) como en el cliente, evitando errores de hidratacion de React / Next.js.
 */
function seededRandom(seed: number): number {
  const x = Math.sin(seed + 1) * 10000;
  return x - Math.floor(x);
}

function generateFrames(): Frame[] {
  const frames: Frame[] = [];
  for (let epoch = 1; epoch <= 50; epoch++) {
    const progress = epoch / 50;
    // Las semillas son constantes: mismo resultado en servidor y cliente
    const dLoss = 0.7 * Math.exp(-progress * 0.5) + 0.3 + seededRandom(epoch * 1.1) * 0.1 - 0.05;
    const gLoss = 2.5 * Math.exp(-progress * 0.8) + 0.8 + seededRandom(epoch * 2.3) * 0.15 - 0.075;
    const quality = Math.min(1, progress * 1.2);
    frames.push({ epoch, dLoss, gLoss, quality });
  }
  return frames;
}

function drawPixelImage(
  canvas: HTMLCanvasElement,
  quality: number,
  epoch: number
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const size = canvas.width;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, size, size);

  const cellSize = size / 10;
  // A mayor quality, mas "coherente" es el patron generado
  for (let row = 0; row < 10; row++) {
    for (let col = 0; col < 10; col++) {
      const x = col / 9;
      const y = row / 9;
      const seed = epoch * 0.01;

      // Ruido determinista: misma semilla = mismo valor en servidor y cliente
      const noise = seededRandom(row * 100 + col + epoch * 13);
      const signal =
        Math.abs(Math.sin(x * Math.PI * 3 + seed)) *
        Math.abs(Math.cos(y * Math.PI * 2 + seed * 0.5));

      const intensity = noise * (1 - quality) + signal * quality;
      const gray = Math.round(Math.max(0, Math.min(1, intensity)) * 220);

      ctx.fillStyle = `rgb(${gray},${Math.round(gray * 0.85)},${Math.round(gray * 1.1)})`;
      ctx.fillRect(col * cellSize, row * cellSize, cellSize - 1, cellSize - 1);
    }
  }
}

const FRAMES = generateFrames();

export default function TrainingAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const frame = FRAMES[currentFrame];

  useEffect(() => {
    if (canvasRef.current && frame) {
      drawPixelImage(canvasRef.current, frame.quality, frame.epoch);
    }
  }, [frame]);

  const play = () => {
    setIsPlaying(true);
    intervalRef.current = setInterval(() => {
      setCurrentFrame((prev) => {
        if (prev >= FRAMES.length - 1) {
          setIsPlaying(false);
          if (intervalRef.current) clearInterval(intervalRef.current);
          return prev;
        }
        return prev + 1;
      });
    }, 120);
  };

  const pause = () => {
    setIsPlaying(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  const reset = () => {
    pause();
    setCurrentFrame(0);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const maxDLoss = Math.max(...FRAMES.map((f) => f.dLoss));
  const maxGLoss = Math.max(...FRAMES.map((f) => f.gLoss));

  return (
    <section className="py-24 px-6 bg-slate-950">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Animacion del entrenamiento
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Observa como evolucionan las perdidas del Generador y el
            Discriminador a lo largo de las epocas, y como la calidad de las
            muestras generadas mejora progresivamente.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-10 items-center">
          {/* Imagen generada */}
          <div className="flex flex-col items-center gap-4">
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={240}
                height={240}
                className="rounded-xl border border-slate-700 shadow-xl"
                style={{ imageRendering: "pixelated" }}
              />
              <div className="absolute top-3 left-3 bg-slate-900/80 rounded-md px-2 py-1 text-xs text-slate-400">
                Epoca {frame.epoch}
              </div>
              <div className="absolute bottom-3 right-3 bg-slate-900/80 rounded-md px-2 py-1 text-xs">
                <span className="text-emerald-400">
                  Calidad: {Math.round(frame.quality * 100)}%
                </span>
              </div>
            </div>

            {/* Controles */}
            <div className="flex gap-3">
              <button
                onClick={reset}
                className="px-4 py-2 rounded-lg border border-slate-700 text-slate-400 hover:text-white text-sm transition-colors"
              >
                Reiniciar
              </button>
              <button
                onClick={isPlaying ? pause : play}
                className={`px-6 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isPlaying
                    ? "bg-red-600/20 border border-red-600 text-red-400"
                    : "bg-violet-600 text-white hover:bg-violet-500"
                }`}
              >
                {isPlaying ? "Pausar" : currentFrame >= FRAMES.length - 1 ? "Repetir" : "Reproducir"}
              </button>
            </div>

            {/* Barra de progreso */}
            <div className="w-full">
              <input
                type="range"
                min="0"
                max={FRAMES.length - 1}
                value={currentFrame}
                onChange={(e) => {
                  pause();
                  setCurrentFrame(parseInt(e.target.value));
                }}
                className="w-full accent-violet-500"
              />
              <div className="flex justify-between text-xs text-slate-600 mt-1">
                <span>Epoca 1</span>
                <span>Epoca {FRAMES.length}</span>
              </div>
            </div>
          </div>

          {/* Grafica de perdidas */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
            <p className="text-sm font-medium text-slate-300 mb-4">
              Curva de perdidas
            </p>
            <svg viewBox="0 0 300 180" className="w-full">
              {/* Ejes */}
              <line x1="30" y1="10" x2="30" y2="155" stroke="#334155" strokeWidth="1" />
              <line x1="30" y1="155" x2="290" y2="155" stroke="#334155" strokeWidth="1" />

              {/* Etiquetas */}
              <text x="15" y="15" fill="#64748b" fontSize="8" textAnchor="middle">Alto</text>
              <text x="15" y="157" fill="#64748b" fontSize="8" textAnchor="middle">Bajo</text>
              <text x="160" y="170" fill="#64748b" fontSize="8" textAnchor="middle">Epocas de entrenamiento</text>

              {/* Linea del Discriminador */}
              <polyline
                points={FRAMES.slice(0, currentFrame + 1)
                  .map((f, i) => {
                    const x = 30 + (i / (FRAMES.length - 1)) * 260;
                    const y = 155 - (f.dLoss / maxDLoss) * 135;
                    return `${x},${y}`;
                  })
                  .join(" ")}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="1.5"
              />

              {/* Linea del Generador */}
              <polyline
                points={FRAMES.slice(0, currentFrame + 1)
                  .map((f, i) => {
                    const x = 30 + (i / (FRAMES.length - 1)) * 260;
                    const y = 155 - (f.gLoss / maxGLoss) * 135;
                    return `${x},${y}`;
                  })
                  .join(" ")}
                fill="none"
                stroke="#a78bfa"
                strokeWidth="1.5"
              />

              {/* Linea de referencia equilibrio */}
              <line x1="30" y1="100" x2="290" y2="100" stroke="#1e293b" strokeWidth="1" strokeDasharray="4,3" />
              <text x="292" y="103" fill="#334155" fontSize="7">Equilibrio</text>
            </svg>

            <div className="flex gap-4 mt-3 text-xs">
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-1.5 rounded bg-blue-500" />
                <span className="text-slate-400">Perdida Discriminador: {frame.dLoss.toFixed(3)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-4 h-1.5 rounded bg-violet-400" />
                <span className="text-slate-400">Perdida Generador: {frame.gLoss.toFixed(3)}</span>
              </div>
            </div>

            <div className="mt-4 p-3 rounded-lg bg-slate-800/50 text-xs text-slate-500 leading-relaxed">
              {frame.quality < 0.3
                ? "Fase inicial: el Generador produce ruido. El Discriminador detecta facilmente todos los falsos."
                : frame.quality < 0.6
                ? "Fase media: el Generador empieza a capturar patrones basicos. Las perdidas comienzan a estabilizarse."
                : frame.quality < 0.85
                ? "Fase avanzada: las muestras generadas son cada vez mas coherentes. La competencia entre redes es intensa."
                : "Fase de convergencia: el Generador produce muestras de alta calidad. El Discriminador se acerca al 50%."}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
