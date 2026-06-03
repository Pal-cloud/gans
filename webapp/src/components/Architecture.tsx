"use client";

export default function Architecture() {
  return (
    <section className="py-24 px-6 bg-slate-950">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Arquitectura de una GAN
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Diagrama del flujo de datos entre el Generador, el Discriminador y
            los datos reales.
          </p>
        </div>

        {/* Diagrama SVG */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 overflow-x-auto">
          <svg
            viewBox="0 0 800 320"
            className="w-full max-w-3xl mx-auto"
            aria-label="Diagrama de arquitectura GAN"
          >
            {/* Ruido latente */}
            <rect x="10" y="120" width="110" height="60" rx="10" fill="#1e1b4b" stroke="#6d28d9" strokeWidth="1.5" />
            <text x="65" y="145" textAnchor="middle" fill="#c4b5fd" fontSize="11" fontWeight="600">Espacio</text>
            <text x="65" y="161" textAnchor="middle" fill="#c4b5fd" fontSize="11" fontWeight="600">Latente (z)</text>

            {/* Flecha ruido -> generador */}
            <path d="M120 150 L170 150" stroke="#6d28d9" strokeWidth="1.5" markerEnd="url(#arrowV)" />

            {/* Generador */}
            <rect x="170" y="100" width="130" height="100" rx="12" fill="#1e1b4b" stroke="#7c3aed" strokeWidth="2" />
            <text x="235" y="140" textAnchor="middle" fill="#a78bfa" fontSize="13" fontWeight="700">GENERADOR</text>
            <text x="235" y="158" textAnchor="middle" fill="#7c3aed" fontSize="10">G(z)</text>
            <text x="235" y="175" textAnchor="middle" fill="#64748b" fontSize="9">Red Neuronal</text>

            {/* Flecha generador -> discriminador */}
            <path d="M300 150 L380 150" stroke="#818cf8" strokeWidth="1.5" strokeDasharray="5,3" markerEnd="url(#arrowI)" />
            <text x="340" y="143" textAnchor="middle" fill="#64748b" fontSize="9">Muestra</text>
            <text x="340" y="155" textAnchor="middle" fill="#64748b" fontSize="9">generada</text>

            {/* Datos reales */}
            <rect x="380" y="20" width="120" height="55" rx="10" fill="#052e16" stroke="#16a34a" strokeWidth="1.5" />
            <text x="440" y="43" textAnchor="middle" fill="#86efac" fontSize="11" fontWeight="600">Datos</text>
            <text x="440" y="59" textAnchor="middle" fill="#86efac" fontSize="11" fontWeight="600">Reales</text>

            {/* Flecha datos reales -> discriminador */}
            <path d="M440 75 L440 110" stroke="#16a34a" strokeWidth="1.5" markerEnd="url(#arrowG)" />

            {/* Discriminador */}
            <rect x="380" y="100" width="130" height="100" rx="12" fill="#172554" stroke="#2563eb" strokeWidth="2" />
            <text x="445" y="140" textAnchor="middle" fill="#93c5fd" fontSize="13" fontWeight="700">DISCRIMINADOR</text>
            <text x="445" y="158" textAnchor="middle" fill="#2563eb" fontSize="10">D(x)</text>
            <text x="445" y="175" textAnchor="middle" fill="#64748b" fontSize="9">Red Neuronal</text>

            {/* Flecha discriminador -> decision */}
            <path d="M510 150 L580 150" stroke="#2563eb" strokeWidth="1.5" markerEnd="url(#arrowB)" />

            {/* Decision */}
            <rect x="580" y="120" width="110" height="60" rx="10" fill="#172554" stroke="#3b82f6" strokeWidth="1.5" />
            <text x="635" y="145" textAnchor="middle" fill="#93c5fd" fontSize="11" fontWeight="600">Real / Falso</text>
            <text x="635" y="161" textAnchor="middle" fill="#3b82f6" fontSize="10">P(real) in [0,1]</text>

            {/* Flecha retroalimentacion al generador */}
            <path d="M635 180 L635 280 L235 280 L235 200" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="6,3" markerEnd="url(#arrowA)" />
            <text x="430" y="298" textAnchor="middle" fill="#f59e0b" fontSize="10">Gradiente de error</text>
            <text x="430" y="310" textAnchor="middle" fill="#92400e" fontSize="9">(backpropagation)</text>

            {/* Definicion de marcadores de flecha */}
            <defs>
              <marker id="arrowV" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#6d28d9" />
              </marker>
              <marker id="arrowI" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#818cf8" />
              </marker>
              <marker id="arrowG" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#16a34a" />
              </marker>
              <marker id="arrowB" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#2563eb" />
              </marker>
              <marker id="arrowA" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#f59e0b" />
              </marker>
            </defs>
          </svg>
        </div>

        {/* Leyenda */}
        <div className="grid md:grid-cols-3 gap-4 mt-8">
          {[
            { color: "violet", label: "Generador", desc: "Convierte ruido en muestras sinteticas" },
            { color: "blue", label: "Discriminador", desc: "Distingue muestras reales de generadas" },
            { color: "amber", label: "Gradiente", desc: "La senal de error que guia el aprendizaje" },
          ].map((item) => (
            <div
              key={item.label}
              className="flex gap-3 p-4 rounded-xl bg-slate-900 border border-slate-800"
            >
              <div
                className={`w-3 h-3 rounded-full mt-1 flex-shrink-0 ${
                  item.color === "violet"
                    ? "bg-violet-500"
                    : item.color === "blue"
                    ? "bg-blue-500"
                    : "bg-amber-500"
                }`}
              />
              <div>
                <p className="text-white font-medium text-sm">{item.label}</p>
                <p className="text-slate-500 text-xs mt-0.5">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
