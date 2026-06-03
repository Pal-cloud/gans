import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GANs - Redes Generativas Adversarias",
  description:
    "Pildora formativa interactiva sobre Redes Generativas Adversarias. Aprende que son las GANs, como funcionan y donde se aplican mediante visualizaciones interactivas.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body className="bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
