"use client";

export default function Footer() {
  return (
    <footer
      className="relative px-6 py-16 overflow-hidden"
      style={{ borderTop: "1px solid var(--rule)" }}
    >
      {/* watermark card-stack motif */}
      <div className="absolute right-8 bottom-8 opacity-5 pointer-events-none select-none" aria-hidden="true">
        <svg width="120" height="100" viewBox="0 0 120 100" fill="none">
          <rect x="12" y="40" width="96" height="60" rx="2" stroke="#C9A227" strokeWidth="2" />
          <rect x="24" y="22" width="72" height="60" rx="2" stroke="#C9A227" strokeWidth="2" />
          <rect x="36" y="4" width="48" height="60" rx="2" stroke="#C9A227" strokeWidth="2" />
        </svg>
      </div>

      <div className="mx-auto max-w-content relative z-10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6 mb-8">
          <span
            className="font-display text-xl text-gold font-semibold"
            style={{ fontOpticalSizing: "auto" } as React.CSSProperties}
          >
            PageKV
          </span>
          <div className="flex gap-6">
            <a
              href="https://github.com/VaibhavD2006/PageKV"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-xs transition-colors"
              style={{ color: "var(--paper-muted)" }}
              onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--paper)")}
              onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--paper-muted)")}
            >
              GitHub
            </a>
            <a
              href="https://github.com/VaibhavD2006/PageKV/blob/master/README.md"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono text-xs transition-colors"
              style={{ color: "var(--paper-muted)" }}
              onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--paper)")}
              onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--paper-muted)")}
            >
              Docs
            </a>
            <a
              href="#benchmarks"
              className="font-mono text-xs transition-colors"
              style={{ color: "var(--paper-muted)" }}
              onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--paper)")}
              onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = "var(--paper-muted)")}
            >
              Benchmarks
            </a>
          </div>
        </div>
        <div className="font-mono text-xs" style={{ color: "var(--rule)" }}>
          MIT License · Open source · Benchmarks in progress
        </div>
      </div>
    </footer>
  );
}
