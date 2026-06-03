"use client";

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-slate-950 px-6">
      {/* Fondo decorativo con gradientes */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-violet-900/20 blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 rounded-full bg-blue-900/20 blur-3xl" />
        <div className="absolute -bottom-20 left-1/3 w-96 h-96 rounded-full bg-indigo-900/15 blur-3xl" />
      </div>

      {/* Grid de puntos de fondo */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage:
            "radial-gradient(circle, #475569 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />

      {/* Contenido principal */}
      <div className="relative z-10 max-w-4xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 text-sm font-medium mb-8">
          <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
          Pildora formativa interactiva
        </div>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
          <span className="text-white">Redes</span>{" "}
          <span className="bg-gradient-to-r from-violet-400 to-blue-400 bg-clip-text text-transparent">
            Generativas
          </span>
          <br />
          <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Adversarias
          </span>
        </h1>

        <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-4 leading-relaxed">
          Aprende como funcionan las GANs desde cero, sin necesidad de conocer
          programacion. Explora los conceptos de forma visual e interactiva.
        </p>

        <p className="text-sm text-slate-500 mb-12">
          Basado en Goodfellow et al. (2014) &middot; Propuesta por Ian
          Goodfellow
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <a
            href="#concepto"
            className="px-8 py-4 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-semibold transition-colors duration-200"
          >
            Empezar a aprender
          </a>
          <a
            href="#espacio-latente"
            className="px-8 py-4 rounded-xl border border-slate-700 hover:border-slate-500 text-slate-300 hover:text-white font-semibold transition-colors duration-200"
          >
            Explorar interactivo
          </a>
        </div>
      </div>

      {/* Indicador de scroll */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-slate-600">
        <span className="text-xs uppercase tracking-widest">Desplazar</span>
        <div className="w-px h-12 bg-gradient-to-b from-slate-600 to-transparent" />
      </div>
    </section>
  );
}
