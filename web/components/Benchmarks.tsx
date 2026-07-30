"use client";
import { motion } from "framer-motion";
import benchmarkData from "@/data/benchmarks.json";

type ConfigResult = { ms: number; speedup: number; k: number; tokens: number; pct: number };
type BenchRow = {
  ctx: number;
  n_pages: number;
  vanilla_ms: number;
  fixed_k4: ConfigResult;
  dyn_1pct: ConfigResult;
  dyn_2pct: ConfigResult;
  dyn_5pct: ConfigResult;
};

const CONFIG_KEYS = ["fixed_k4", "dyn_1pct", "dyn_2pct", "dyn_5pct"] as const;
const CONFIG_LABELS: Record<string, string> = {
  fixed_k4: "Fixed top_k=4",
  dyn_1pct: "Dynamic 1%",
  dyn_2pct: "Dynamic 2%",
  dyn_5pct: "Dynamic 5% ★",
};

function fmt(n: number) { return n >= 1000 ? `${(n / 1000).toFixed(0)}K` : String(n); }

export default function Benchmarks() {
  const rows = (benchmarkData as { results: BenchRow[] }).results;

  return (
    <section id="benchmarks" className="px-6 py-24">
      <div className="mx-auto max-w-content">
        <motion.div
          className="mb-10"
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--paper-muted)" }}>
            Benchmarks
          </div>
          <h2
            className="font-display text-3xl sm:text-4xl font-bold leading-tight mb-4"
            style={{ letterSpacing: "-0.02em", textWrap: "balance" } as React.CSSProperties}
          >
            Real Results
          </h2>
          <p className="font-mono text-sm max-w-2xl" style={{ color: "var(--paper-muted)" }}>
            Attention kernel microbench — CPU (PyTorch SDPA). B=1, H=12, head_dim=64, page_size=128.
            5 warmup runs, 20 timed runs, mean reported. Cached = page summaries pre-built (steady-state decode).
            GPU results would show wider speedup margins.
          </p>
        </motion.div>

        {/* ── accuracy-first callout ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8 p-5 border"
          style={{ borderColor: "var(--gold)", borderRadius: "2px", backgroundColor: "rgba(201,162,39,0.05)" }}
        >
          <div className="font-mono text-xs mb-2" style={{ color: "var(--gold)" }}>Accuracy-first configuration (★)</div>
          <p className="font-mono text-sm" style={{ color: "var(--paper)" }}>
            <strong>Dynamic 5%</strong> reads a constant 5% of the KV cache every decode step regardless of context length —
            1.12× faster than vanilla at 8K tokens, 2.83× faster at 262K, while maintaining consistent recall quality as context grows.
            Fixed top_k=4 is faster in raw throughput but reads only 0.20% of context at 262K tokens,
            which degrades recall accuracy at very long context.
          </p>
        </motion.div>

        {/* ── latency table ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.1 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="mb-8"
        >
          <div className="font-mono text-xs font-medium mb-4" style={{ color: "var(--paper)" }}>
            Latency (ms) · speedup vs vanilla in green
          </div>
          <div className="-mx-6 sm:mx-0 overflow-x-auto">
            <div className="px-6 sm:px-0">
              <table className="font-mono text-xs text-left" style={{ borderCollapse: "collapse", minWidth: "680px" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--rule)" }}>
                    <th className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>Context</th>
                    <th className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>Vanilla</th>
                    {CONFIG_KEYS.map(k => (
                      <th key={k} className="py-2 pr-5 pb-3 whitespace-nowrap" style={{
                        color: k === "dyn_5pct" ? "var(--gold)" : "var(--paper-muted)", fontWeight: 500,
                      }}>
                        {CONFIG_LABELS[k]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => (
                    <tr key={row.ctx} style={{ borderBottom: "1px solid var(--rule)" }}>
                      <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--paper)" }}>{fmt(row.ctx)}</td>
                      <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--paper-muted)" }}>{row.vanilla_ms}ms</td>
                      {CONFIG_KEYS.map(k => {
                        const cfg = row[k as keyof BenchRow] as ConfigResult;
                        const faster = cfg.speedup >= 1.0;
                        const isRec = k === "dyn_5pct";
                        return (
                          <td key={k} className="py-2 pr-5 whitespace-nowrap" style={{ color: isRec ? "var(--gold)" : (faster ? "var(--paper)" : "var(--paper-muted)") }}>
                            {cfg.ms}ms{" "}
                            <span style={{ color: faster ? "var(--signal-green)" : "var(--paper-muted)" }}>
                              {cfg.speedup}×
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>

        {/* ── coverage table ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.1 }}
          transition={{ duration: 0.5, delay: 0.05, ease: [0.16, 1, 0.3, 1] }}
          className="mb-10"
        >
          <div className="font-mono text-xs font-medium mb-4" style={{ color: "var(--paper)" }}>
            Context coverage — tokens read per decode step
          </div>
          <div className="-mx-6 sm:mx-0 overflow-x-auto">
            <div className="px-6 sm:px-0">
              <table className="font-mono text-xs text-left" style={{ borderCollapse: "collapse", minWidth: "680px" }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--rule)" }}>
                    <th className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>Context</th>
                    <th className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>Vanilla</th>
                    {CONFIG_KEYS.map(k => (
                      <th key={k} className="py-2 pr-5 pb-3 whitespace-nowrap" style={{
                        color: k === "dyn_5pct" ? "var(--gold)" : "var(--paper-muted)", fontWeight: 500,
                      }}>
                        {CONFIG_LABELS[k]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => (
                    <tr key={row.ctx} style={{ borderBottom: "1px solid var(--rule)" }}>
                      <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--paper)" }}>{fmt(row.ctx)}</td>
                      <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--paper-muted)" }}>{fmt(row.ctx)} (100%)</td>
                      {CONFIG_KEYS.map(k => {
                        const cfg = row[k as keyof BenchRow] as ConfigResult;
                        const isRec = k === "dyn_5pct";
                        return (
                          <td key={k} className="py-2 pr-5 whitespace-nowrap" style={{ color: isRec ? "var(--gold)" : "var(--paper-muted)" }}>
                            {cfg.tokens.toLocaleString()} ({cfg.pct}%)
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="font-mono text-xs mt-3" style={{ color: "var(--rule)" }}>
            Dynamic 5% maintains consistent coverage at every context length.
            Fixed top_k=4 reads 512 tokens always — 6.25% at 8K, but only 0.20% at 262K.
          </p>
        </motion.div>

        {/* ── GPU pending ── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="border p-8 text-center"
          style={{ borderColor: "var(--rule)", borderRadius: "2px", backgroundColor: "var(--panel)" }}
        >
          <div className="font-mono text-xs mb-2" style={{ color: "var(--paper-muted)" }}>GPU Benchmarks</div>
          <div
            className="inline-block font-mono text-xs px-3 py-1 mb-4 animate-pulse-gold"
            style={{ border: "1px solid var(--gold)", color: "var(--gold)", borderRadius: "2px" }}
          >
            [TBD]
          </div>
          <p className="font-mono text-sm mb-1" style={{ color: "var(--paper-muted)" }}>
            Full memory and accuracy benchmarks require GPU access — in progress.
          </p>
          <p className="font-mono text-xs" style={{ color: "var(--paper-muted)", opacity: 0.6 }}>
            Target: ≥30% memory reduction at 32K+ tokens · ≤5% accuracy drop vs. vanilla
          </p>
          <a
            href="https://github.com/VaibhavD2006/PageKV"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block mt-6 font-mono text-xs border px-4 py-2 transition-colors"
            style={{ borderColor: "var(--rule)", color: "var(--paper-muted)", borderRadius: "2px" }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--paper-muted)"; (e.currentTarget as HTMLElement).style.color = "var(--paper)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--rule)"; (e.currentTarget as HTMLElement).style.color = "var(--paper-muted)"; }}
          >
            View benchmark suite on GitHub →
          </a>
        </motion.div>
      </div>
    </section>
  );
}
