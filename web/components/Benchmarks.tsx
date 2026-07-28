"use client";
import benchmarkData from "@/data/benchmarks.json";

const columns = ["Context Length", "Memory (Vanilla)", "Memory (PageKV)", "Accuracy (Vanilla)", "Accuracy (PageKV)", "Latency (Vanilla)", "Latency (PageKV)"];

export default function Benchmarks() {
  const isPending = benchmarkData.meta.status === "pending" || benchmarkData.results.length === 0;

  return (
    <section id="benchmarks" className="px-6 py-24">
      <div className="mx-auto max-w-content">
        <div className="mb-10">
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--paper-muted)" }}>
            Benchmarks
          </div>
          <h2
            className="font-display text-3xl sm:text-4xl font-bold leading-tight"
            style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em" } as React.CSSProperties}
          >
            Results
          </h2>
        </div>

        {isPending ? (
          <div
            className="border p-8 text-center"
            style={{ borderColor: "var(--rule)", borderRadius: "2px", backgroundColor: "var(--panel)" }}
          >
            <div
              className="inline-block font-mono text-xs px-3 py-1 mb-4"
              style={{ border: "1px solid var(--gold)", color: "var(--gold)", borderRadius: "2px" }}
            >
              [TBD]
            </div>
            <p className="font-mono text-sm mb-2" style={{ color: "var(--paper-muted)" }}>
              Benchmarks in progress — see GitHub for current numbers.
            </p>
            <p className="font-mono text-xs" style={{ color: "var(--paper-muted)", opacity: 0.6 }}>
              Target metrics (from PRD): ≥30% memory reduction at 32K+ tokens · ≤5% accuracy drop vs. vanilla
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
                      {columns.map((col) => (
                        <th key={col} className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {["4K", "16K", "32K", "64K"].map((ctx) => (
                      <tr key={ctx} style={{ borderBottom: "1px solid var(--rule)" }}>
                        <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--paper-muted)" }}>{ctx}</td>
                        {columns.slice(1).map((col) => (
                          <td key={col} className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--rule)" }}>—</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="font-mono text-xs text-left" style={{ borderCollapse: "collapse", minWidth: "640px" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--rule)" }}>
                  {columns.map((col) => (
                    <th key={col} className="py-2 pr-5 pb-3 whitespace-nowrap" style={{ color: "var(--paper-muted)", fontWeight: 500 }}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(benchmarkData.results as Record<string, string | number>[]).map((row, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--rule)" }}>
                    <td className="py-2 pr-5 whitespace-nowrap">{row.ctx_len}</td>
                    <td className="py-2 pr-5 whitespace-nowrap">{row.memory_vanilla_gb} GB</td>
                    <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--signal-green)" }}>{row.memory_pagekv_gb} GB</td>
                    <td className="py-2 pr-5 whitespace-nowrap">{row.accuracy_vanilla}</td>
                    <td className="py-2 pr-5 whitespace-nowrap">{row.accuracy_pagekv}</td>
                    <td className="py-2 pr-5 whitespace-nowrap">{row.latency_vanilla_ms} ms</td>
                    <td className="py-2 pr-5 whitespace-nowrap" style={{ color: "var(--signal-green)" }}>{row.latency_pagekv_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
