"use client";
import { useState } from "react";

export default function Quickstart() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard?.writeText("pip install pagekv");
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <section id="install" className="px-6 py-24">
      <div className="mx-auto max-w-content text-center">
        <div className="font-mono text-xs tracking-widest uppercase mb-4" style={{ color: "var(--paper-muted)" }}>
          Get Started
        </div>
        <h2
          className="font-display text-3xl sm:text-4xl font-bold mb-6 leading-tight"
          style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em" } as React.CSSProperties}
        >
          One command. Drop-in compatible.
        </h2>
        <p className="text-base mb-10 max-w-lg mx-auto" style={{ color: "var(--paper-muted)" }}>
          Works with any HuggingFace model. The generation API is unchanged — swap in PageKV and measure on your own workload.
        </p>

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <a
            href="#docs"
            className="font-mono text-base px-8 py-3.5 min-h-[48px] transition-colors flex items-center justify-center"
            style={{ backgroundColor: "var(--gold)", color: "var(--void)", borderRadius: "2px" }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.backgroundColor = "var(--gold-bright)")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.backgroundColor = "var(--gold)")}
          >
            View Docs
          </a>
          <a
            href="https://github.com/VaibhavD2006/PageKV"
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-base px-8 py-3.5 min-h-[48px] border transition-colors flex items-center justify-center"
            style={{ borderColor: "var(--rule)", color: "var(--paper-muted)", borderRadius: "2px" }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.borderColor = "var(--paper-muted)";
              (e.currentTarget as HTMLElement).style.color = "var(--paper)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.borderColor = "var(--rule)";
              (e.currentTarget as HTMLElement).style.color = "var(--paper-muted)";
            }}
          >
            View on GitHub →
          </a>
        </div>
      </div>
    </section>
  );
}
