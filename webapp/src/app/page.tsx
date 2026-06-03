import Hero from "@/components/Hero";
import ConceptExplainer from "@/components/ConceptExplainer";
import Architecture from "@/components/Architecture";
import TrainingAnimation from "@/components/TrainingAnimation";
import LatentSpaceExplorer from "@/components/LatentSpaceExplorer";
import DomainShowcase from "@/components/DomainShowcase";

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
