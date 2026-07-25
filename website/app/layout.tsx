import { RootProvider } from "fumadocs-ui/provider/next";
import { Manrope } from "next/font/google";
import type { ReactNode } from "react";
import "./global.css";

const manrope = Manrope({ subsets: ["latin"] });

export const metadata = {
  title: {
    template: "%s — Superfermion docs",
    default: "Superfermion docs — Quantum Circuit Simulator",
  },
  description:
    "Documentation for Superfermion: a high-performance quantum-circuit simulator with 12+ backends, adjoint differentiation, MPS tensor networks, and a Rust SIMD acceleration core.",
  keywords: [
    "Superfermion documentation",
    "quantum circuit simulator",
    "quantum computing Python",
    "MPS tensor network",
    "adjoint differentiation",
    "quantum error correction",
    "PyO3 quantum simulator",
  ],
  openGraph: {
    title: "Superfermion docs — Quantum Circuit Simulator",
    description:
      "Complete documentation for Superfermion: installation, backends, tutorials, API reference, and CLI.",
    type: "website",
    url: "https://superfermion.com",
    siteName: "Superfermion docs",
  },
  alternates: {
    canonical: "https://superfermion.com",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={manrope.className} suppressHydrationWarning>
      <body className="flex min-h-screen flex-col">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
