"use client";
import { motion } from "framer-motion";

export default function Problem() {
  return (
    <section id="problem" className="px-6 py-24">
      <div className="mx-auto max-w-content">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="font-mono text-xs tracking-widest uppercase mb-4" style={{ color: "var(--paper-muted)" }}>
              The Problem
            </div>
            <h2
              className="font-display text-3xl sm:text-4xl font-bold mb-5 leading-tight"
              style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em", textWrap: "balance" } as React.CSSProperties}
            >
              KV cache grows with every token you process.
            </h2>
            <div className="space-y-3 text-base leading-relaxed" style={{ color: "var(--paper-muted)" }}>
              <p>
                Every token a transformer processes gets a key and value stored in memory. At 32K tokens, that&apos;s gigabytes of state — most of it irrelevant to any single query step.
              </p>
              <p>
                Standard attention reads the entire cache on every decode step. At long context, the GPU spends its time moving data it won&apos;t use — not computing.
              </p>
            </div>
          </motion.div>

          <div className="space-y-5">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            >
              <div className="font-mono text-xs mb-2" style={{ color: "var(--paper-muted)" }}>
                Context grows →
              </div>
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: "100%" }}
                viewport={{ once: true, amount: 0.5 }}
                transition={{ duration: 0.85, ease: "easeOut", delay: 0.1 }}
                className="h-9 flex items-center pl-3"
                style={{ backgroundColor: "rgba(201,162,39,0.25)", border: "1px solid var(--gold)", borderRadius: "2px" }}
              >
                <span className="font-mono text-xs" style={{ color: "var(--gold)" }}>
                  100K tokens · KV cache grows linearly
                </span>
              </motion.div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.5, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
            >
              <div className="font-mono text-xs mb-2" style={{ color: "var(--signal-green)" }}>
                What one decode step actually needs →
              </div>
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: "22%" }}
                viewport={{ once: true, amount: 0.5 }}
                transition={{ duration: 0.85, ease: "easeOut", delay: 0.3 }}
                className="h-9 flex items-center pl-3"
                style={{ backgroundColor: "rgba(111,191,139,0.2)", border: "1px solid var(--signal-green)", borderRadius: "2px" }}
              >
                <span className="font-mono text-xs" style={{ color: "var(--signal-green)", whiteSpace: "nowrap" }}>
                  top-4 pages
                </span>
              </motion.div>
            </motion.div>

            <p className="font-mono text-xs" style={{ color: "var(--rule)" }}>
              Illustrative ratio. Exact savings depend on model, page_size, and top_k_pages.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}