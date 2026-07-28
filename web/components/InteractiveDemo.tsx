"use client";
import { useState, useRef } from "react";
import { useReducedMotion, motion, AnimatePresence } from "framer-motion";

type Phase = "idle" | "paging" | "summarizing" | "routing" | "done";

const NUM_PAGES = 12;
const SELECTED_PAGES = [2, 5, 9];

export default function InteractiveDemo() {
  const reduced = useReducedMotion() ?? false;
  const [phase, setPhase] = useState<Phase>("idle");
  const [memStart] = useState(4.2);
  const [memEnd] = useState(1.6);
  const [currentMem, setCurrentMem] = useState(4.2);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clear = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  const reset = () => {
    clear();
    setPhase("idle");
    setCurrentMem(memStart);
  };

  const run = () => {
    if (phase !== "idle") return;
    if (reduced) {
      setPhase("done");
      setCurrentMem(memEnd);
      return;
    }

    setPhase("paging");
    timerRef.current = setTimeout(() => {
      setPhase("summarizing");
      timerRef.current = setTimeout(() => {
        setPhase("routing");
        timerRef.current = setTimeout(() => {
          setPhase("done");
          // tick memory counter down
          let current = memStart;
          const step = (memStart - memEnd) / 20;
          const tick = () => {
            current = Math.max(memEnd, current - step);
            setCurrentMem(parseFloat(current.toFixed(2)));
            if (current > memEnd) timerRef.current = setTimeout(tick, 60);
          };
          tick();
        }, 1200);
      }, 1200);
    }, 1000);
  };

  const phaseLabel: Record<Phase, string> = {
    idle: "Ready",
    paging: "Grouping tokens into pages…",
    summarizing: "Compressing each page to a summary key…",
    routing: "Query routing to top-K pages…",
    done: "Attention complete.",
  };

  return (
    <section id="demo" className="px-6 py-24">
      <div className="mx-auto max-w-content">
        <div className="mb-10">
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--paper-muted)" }}>
            Interactive Demo
          </div>
          <h2
            className="font-display text-3xl sm:text-4xl font-bold leading-tight"
            style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em" } as React.CSSProperties}
          >
            Watch the orchestration.
          </h2>
        </div>

        <div
          className="border p-6 sm:p-8"
          style={{ borderColor: "var(--rule)", borderRadius: "2px", backgroundColor: "var(--panel)" }}
        >
          {/* Page grid */}
          <div className="grid grid-cols-6 sm:grid-cols-12 gap-2 mb-8">
            {Array.from({ length: NUM_PAGES }, (_, i) => {
              const isSelected = SELECTED_PAGES.includes(i);
              const isSummarized = phase === "summarizing" || phase === "routing" || phase === "done";
              const isHighlighted = (phase === "routing" || phase === "done") && isSelected;

              return (
                <motion.div
                  key={i}
                  animate={
                    isHighlighted
                      ? { borderColor: "var(--gold)", boxShadow: "0 0 10px 2px rgba(201,162,39,0.3)", backgroundColor: "rgba(201,162,39,0.12)" }
                      : isSummarized
                      ? { borderColor: "#2A2A2E", boxShadow: "none", backgroundColor: "var(--panel)" }
                      : { borderColor: "#2A2A2E", boxShadow: "none", backgroundColor: "#16161A" }
                  }
                  transition={{ duration: 0.3, delay: isHighlighted ? i * 0.04 : 0 }}
                  className="border aspect-square flex items-center justify-center"
                  style={{ borderRadius: "2px" }}
                  title={`Page ${i + 1}`}
                >
                  <span className="font-mono text-xs" style={{ color: isHighlighted ? "var(--gold)" : "var(--rule)", fontSize: "10px" }}>
                    {i + 1}
                  </span>
                </motion.div>
              );
            })}
          </div>

          {/* Status + counters */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div className="font-mono text-sm" style={{ color: "var(--paper-muted)" }}>
              <AnimatePresence mode="wait">
                <motion.span
                  key={phase}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.2 }}
                >
                  {phaseLabel[phase]}
                </motion.span>
              </AnimatePresence>
            </div>
            {phase === "done" && (
              <div className="font-mono text-sm flex flex-col sm:flex-row gap-2 sm:gap-6">
                <span style={{ color: "var(--paper-muted)" }}>
                  Memory:{" "}
                  <span style={{ color: "var(--signal-green)" }}>{currentMem.toFixed(1)} GB</span>
                  <span style={{ color: "var(--rule)", fontSize: "10px" }}> (was {memStart} GB)</span>
                </span>
                <span style={{ color: "var(--paper-muted)" }}>
                  Pages attended:{" "}
                  <span style={{ color: "var(--gold)" }}>{SELECTED_PAGES.length} / {NUM_PAGES}</span>
                </span>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="flex gap-3">
            <button
              onClick={run}
              disabled={phase !== "idle"}
              className="font-mono text-sm px-5 py-2.5 min-h-[44px] transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ backgroundColor: "var(--gold)", color: "var(--void)", borderRadius: "2px" }}
              onMouseEnter={(e) => { if (phase === "idle") (e.currentTarget as HTMLElement).style.backgroundColor = "var(--gold-bright)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = "var(--gold)"; }}
            >
              Run Demo
            </button>
            <button
              onClick={reset}
              className="font-mono text-sm px-5 py-2.5 min-h-[44px] border transition-colors cursor-pointer"
              style={{ borderColor: "var(--rule)", color: "var(--paper-muted)", borderRadius: "2px" }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "var(--paper-muted)"; (e.currentTarget as HTMLElement).style.color = "var(--paper)"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = "var(--rule)"; (e.currentTarget as HTMLElement).style.color = "var(--paper-muted)"; }}
            >
              Reset
            </button>
          </div>
        </div>

        <p className="mt-4 font-mono text-xs" style={{ color: "var(--paper-muted)" }}>
          Simulated for illustration. Run the real benchmark suite locally —{" "}
          <a
            href="#benchmarks"
            style={{ color: "var(--gold)", textDecoration: "underline" }}
          >
            see Benchmarks
          </a>
          . Memory figures are illustrative, not measured.
        </p>
      </div>
    </section>
  );
}
