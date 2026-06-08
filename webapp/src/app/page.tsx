import dynamic from "next/dynamic";

// Todos los componentes interactivos se cargan exclusivamente en el cliente
// (ssr: false) para evitar discrepancias de hidratacion entre servidor y cliente
// que causan el error "Missing ActionQueueContext" del Router de Next.js.
const Hero = dynamic(() => import("@/components/Hero"), { ssr: false });
const ConceptExplainer = dynamic(() => import("@/components/ConceptExplainer"), { ssr: false });
const Architecture = dynamic(() => import("@/components/Architecture"), { ssr: false });
const TrainingAnimation = dynamic(() => import("@/components/TrainingAnimation"), { ssr: false });
const LatentSpaceExplorer = dynamic(() => import("@/components/LatentSpaceExplorer"), { ssr: false });
const DomainShowcase = dynamic(() => import("@/components/DomainShowcase"), { ssr: false });

export default function Home() {
  return (
    <main className="min-h-screen">
      <Hero />
      <ConceptExplainer />
      <Architecture />
      <TrainingAnimation />
      <LatentSpaceExplorer />
      <DomainShowcase />

      <footer className="border-t border-slate-800 py-10 text-center text-slate-500 text-sm">
        <p>Pildora formativa sobre Redes Generativas Adversarias</p>
        <p className="mt-1">
          Basada en Goodfellow et al. (2014) - Generative Adversarial Nets
        </p>
      </footer>
    </main>
  );
}
