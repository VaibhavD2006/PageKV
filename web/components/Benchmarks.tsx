"use client";
import { motion } from "framer-motion";
import benchmarkData from "@/data/benchmarks.json";

const gpuColumns = ["Context Length", "Memory (Vanilla)", "Memory (PageKV)", "Accuracy (Vanilla)", "Accuracy (PageKV)", "Latency (Vanilla)", "Latency (PageKV)"];

type CpuRow = {
  ctx: number;
  vanilla_ms: number;
  cached_ms: number;
  speedup: number;
  pct_read: number;
  note: string;
};

export default function Benchmarks() {
  const data = benchmarkData as {
    cpu_timing?: CpuRow[];
    results: unknown[];
    cpu_config?: { page_size: number; top_k: number };
  };

  const cpuRows: CpuRow[] = data.cpu_timing ?? [];
  const hasCpuData = cpuRows.length > 0;
  const gpuPending = data.results.length === 0;
  const crossoverRow = cpuRows.find((r) => r.note === "crossover");
  const cfg = data.cpu_config;

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
            className="font-display text-3xl sm:text-4xl font-bold leading-tight"
            style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em", textWrap: "balance" } as React.CSSProperties}
          >
            Results
          </h2>
        </motion.div>

        {/* ── CPU timing ── */}
        {hasCpuData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.15 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="mb-10"
          >
            <div className="flex flex-col sm:flex-row sm:items-center gap-3 mb-4">
              <div className="font-mono text-xs font-medium" style={{ color: "var(--paper)" }}>
                CPU Latency — Attention Kernel Microbench
              </div>
              <div
                className="inline-block font-mono text-xs px-2 py-0.5 self-start sm:self-auto"
                style={{ border: "1px solid var(--signal-green)", color: "var(--signal-green)", borderRadius: "2px" }}
              >
                Real data · no GPU required
              </div>
            </div>

            {crossoverRow && (
              <div
                className="mb-5 p-4 border font-mono text-sm"
                style={{ borderColor: "var(--gold)", borderRadius: "2px", backgroundColor: "rgba(201,162,39,0.06)" }}
              >
                <span style={{ color: "var(--gold)" }}>
                  Faster than vanilla from ~{crossoverRow.ctx.toLocaleString()} tokens onward.
                </span>
                {" "}Speedup grows with context: {crossoverRow.speedup}x at {crossoverRow.ctx.toLocaleString()} tokens, rising to{" "}
                <span style={{ color: "var(--gold)" }}>
                  {cpuRows[cpuRows.length - 1].speedup}x at {cpuRows[cpuRows.length - 1].ctx.toLocaleString()} tokens
                </span>
                {" "}— while reading only {cpuRows[cpuRows.length - 1].pct_read}% of the KV cache per step.
                On GPU with hardware attention kernels, the crossover shifts earlier and the slope steepens.
              </div>
            )}

            <div className="-mx-6 sm:mx-0 overflow-x-auto">
              <div className="px-6 sm:px-0">
                <table className="font-mono text-xs text-left" style={{ borderCollapse: "collapse", minWidth: "480px" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--rule)" }}>
                      {["Context", "Vanilla", "PageKV cached", "Speedup", "% cache read"].map((h) => (
                        <th key={h} className="py-2 pr-6 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cpuRows.map((row) => {
                      const isCrossover = row.note === "crossover";
                      const faster = row.speedup > 1.0;
                      return (
                        <tr
                          key={row.ctx}
                          style={{
                            borderBottom: "1px solid var(--rule)",
                            backgroundColor: isCrossover ? "rgba(201,162,39,0.05)" : "transparent",
                          }}
                        >
                          <td className="py-2 pr-6 whitespace-nowrap" style={{ color: isCrossover ? "var(--gold)" : "var(--paper)" }}>
                            {row.ctx.toLocaleString()}
                            {isCrossover && <span className="ml-1" style={{ color: "var(--gold)", fontSize: "10px" }}>▲</span>}
                          </td>
                          <td className="py-2 pr-6 whitespace-nowrap" style={{ color: "var(--paper-muted)" }}>{row.vanilla_ms}ms</td>
                          <td className="py-2 pr-6 whitespace-nowrap" style={{ color: faster ? "var(--paper)" : "var(--paper-muted)" }}>
                            {row.cached_ms}ms
                          </td>
                          <td className="py-2 pr-6 whitespace-nowrap font-medium" style={{ color: faster ? "var(--signal-green)" : "var(--paper-muted)" }}>
                            {faster ? `>${row.speedup}x` : `${row.speedup}x`}
                          </td>
                          <td className="py-2 whitespace-nowrap" style={{ color: "var(--paper-muted)" }}>{row.pct_read}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            {cfg && (
              <p className="font-mono text-xs mt-3" style={{ color: "var(--rule)" }}>
                page_size={cfg.page_size} · top_k={cfg.top_k} · heads=12 · head_dim=64 · PyTorch CPU · cached = pre-built summaries
              </p>
            )}
          </motion.div>
        )}

        {/* ── GPU results (TBD) ── */}
        {gpuPending && (
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
            <p className="font-mono text-sm mb-2" style={{ color: "var(--paper-muted)" }}>
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
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = "var(--paper-muted)";
                (e.currentTarget as HTMLElement).style.color = "var(--paper)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = "var(--rule)";
                (e.currentTarget as HTMLElement).style.color = "var(--paper-muted)";
              }}
            >
              View benchmark suite on GitHub →
            </a>
            <div className="mt-8 -mx-8 overflow-x-auto">
              <div className="px-8">
                <table className="font-mono text-xs text-left" style={{ borderCollapse: "collapse", minWidth: "640px" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--rule)" }}>
                      {gpuColumns.map((col) => (
                        <th key={col} className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {["4K", "16K", "32K", "64K"].map((ctx) => (
                      <tr key={ctx} style={{ borderBottom: "1px solid var(--rule)" }}>
                        <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--paper-muted)" }}>{ctx}</td>
                        {gpuColumns.slice(1).map((col) => (
                          <td key={col} className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--rule)" }}>—</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </section>
  );
}