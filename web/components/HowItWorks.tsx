"use client";
import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const steps = [
  {
    number: "01",
    title: "Page",
    body: "Tokens are grouped into fixed-size pages (default: 128 tokens each). A 100K-token context becomes ~781 pages.",
    svg: (
      <svg width="120" height="48" viewBox="0 0 120 48" fill="none" aria-hidden="true">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <rect key={i} x={i * 18} y={4} width="14" height="40" rx="1" stroke="#2A2A2E" strokeWidth="1" fill="#16161A" />
        ))}
        <rect x="108" y="4" width="12" height="40" rx="1" stroke="#C9A227" strokeWidth="1" fill="#16161A" />
      </svg>
    ),
  },
  {
    number: "02",
    title: "Summarize",
    body: "Each page is compressed to one summary key — a single vector capturing that page's content. Choose mean-pool, max-pool, or a learned MLP summarizer.",
    svg: (
      <svg width="120" height="48" viewBox="0 0 120 48" fill="none" aria-hidden="true">
        {[0, 1, 2].map((i) => (
          <rect key={i} x={i * 16} y={8} width="12" height="32" rx="1" stroke="#2A2A2E" strokeWidth="1" fill="#16161A" />
        ))}
        <path d="M56 24 L70 24" stroke="#2A2A2E" strokeWidth="1" />
        <polygon points="70,20 78,24 70,28" fill="#2A2A2E" />
        <rect x="78" y="8" width="16" height="32" rx="1" stroke="#C9A227" strokeWidth="1.5" fill="rgba(201,162,39,0.1)" />
      </svg>
    ),
  },
  {
    number: "03",
    title: "Route",
    body: "A query scores all page summaries with a dot product. Only the top-K pages (default: 4) get full attention — the rest are skipped entirely.",
    svg: (
      <svg width="120" height="48" viewBox="0 0 120 48" fill="none" aria-hidden="true">
        {[0, 1, 2, 3, 4].map((i) => (
          <rect
            key={i}
            x={i * 22 + 4}
            y={18}
            width="16"
            height="20"
            rx="1"
            stroke={i === 2 ? "#C9A227" : "#2A2A2E"}
            strokeWidth={i === 2 ? 1.5 : 1}
            fill={i === 2 ? "rgba(201,162,39,0.15)" : "#16161A"}
          />
        ))}
        <path d="M49 4 L49 16" stroke="#C9A227" strokeWidth="1" />
        <polygon points="45,16 49,22 53,16" fill="#C9A227" />
        <text x="49" y="8" textAnchor="middle" fontSize="7" fill="#C9A227" fontFamily="IBM Plex Mono">
          query
        </text>
      </svg>
    ),
  },
];

export default function HowItWorks() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="how-it-works" className="px-6 py-24">
      <div className="mx-auto max-w-content" ref={ref}>
        <div className="mb-12">
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--paper-muted)" }}>
            The Mechanism
          </div>
          <h2
            className="font-display text-3xl sm:text-4xl font-bold leading-tight"
            style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em" } as React.CSSProperties}
          >
            Three steps. One pass.
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {steps.map((step, i) => (
            <motion.div
              key={step.number}
              initial={{ opacity: 0, y: 24 }}
              animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
              transition={{ duration: 0.5, delay: i * 0.15 }}
              className="border p-6 flex flex-col gap-4"
              style={{ borderColor: "var(--rule)", borderRadius: "2px", backgroundColor: "var(--panel)" }}
            >
              <div className="font-mono text-xs" style={{ color: "var(--gold)" }}>
                {step.number}
              </div>
              <div className="py-2">{step.svg}</div>
              <h3
                className="font-display text-xl font-bold"
                style={{ fontOpticalSizing: "auto" } as React.CSSProperties}
              >
                {step.title}
              </h3>
              <p className="text-sm leading-relaxed" style={{ color: "var(--paper-muted)" }}>
                {step.body}
              </p>
            </motion.div>
          ))}
        </div>

        <div
          className="mt-8 p-4 border font-mono text-sm"
          style={{ borderColor: "var(--rule)", borderRadius: "2px", color: "var(--paper-muted)" }}
        >
          <span style={{ color: "var(--gold)" }}>Correctness guarantee: </span>
          When <code style={{ color: "var(--paper)" }}>top_k_pages</code> equals the total number of pages, PageKV skips routing and runs full attention — output is numerically identical to vanilla within floating-point tolerance (±1e-4).
        </div>
      </div>
    </section>
  );
}
