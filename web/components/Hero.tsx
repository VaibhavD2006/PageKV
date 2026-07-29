"use client";
import { useReducedMotion, motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";

function CardStack({ reduced }: { reduced: boolean }) {
  const [collapsed, setCollapsed] = useState(reduced);

  useEffect(() => {
    if (reduced) return;
    const t = setTimeout(() => setCollapsed(true), 800);
    return () => clearTimeout(t);
  }, [reduced]);

  const cards = [6, 5, 4, 3, 2, 1, 0];

  return (
    <div className="relative w-64 h-44 mx-auto select-none" aria-hidden="true">
      {/* Ambient glow behind card — always present, brightens when collapsed */}
      <motion.div
        className="absolute inset-0 hero-glow pointer-events-none"
        style={{ borderRadius: "12px", zIndex: 0 }}
        animate={collapsed ? { opacity: 1, scale: 1.2 } : { opacity: 0.3, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />

      <AnimatePresence>
        {!collapsed &&
          cards.map((i) => (
            <motion.div
              key={i}
              className="absolute inset-0 border"
              style={{
                borderColor: "var(--rule)",
                backgroundColor: "var(--panel)",
                borderRadius: "2px",
                top: i * 7,
                zIndex: 10 - i,
              }}
              exit={{
                y: 20,
                opacity: 0,
                scale: 0.9,
                transition: { duration: 0.3, delay: (6 - i) * 0.05 },
              }}
            />
          ))}
      </AnimatePresence>

      <motion.div
        animate={
          collapsed
            ? { boxShadow: "0 0 32px 8px rgba(201,162,39,0.25)", borderColor: "var(--gold)" }
            : { boxShadow: "none", borderColor: "var(--rule)" }
        }
        transition={{ duration: 0.7 }}
        className="absolute inset-0 border flex flex-col items-center justify-center gap-3"
        style={{ backgroundColor: "var(--panel)", borderRadius: "2px", zIndex: 20 }}
      >
        <div className="font-mono text-xs tracking-widest" style={{ color: "var(--gold)" }}>
          PAGE SUMMARY
        </div>
        <div className="flex gap-1.5">
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              animate={collapsed ? { backgroundColor: "var(--gold)" } : { backgroundColor: "var(--rule)" }}
              transition={{ duration: 0.4, delay: 0.1 + i * 0.04 }}
              className="w-5 h-1.5 rounded-sm"
            />
          ))}
        </div>
        <div className="font-mono text-xs" style={{ color: "var(--paper-muted)" }}>
          128 tokens → 1 vector
        </div>
      </motion.div>
    </div>
  );
}

const containerVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1 } },
};

const itemVariants = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] } },
};

export default function Hero() {
  const reduced = useReducedMotion() ?? false;

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-20 pb-24 text-center overflow-hidden">
      {/* Grid texture */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(var(--rule) 1px, transparent 1px), linear-gradient(90deg, var(--rule) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
          opacity: 0.18,
        }}
      />

      {/* Radial hero glow — centered on card-stack */}
      <div
        className="absolute pointer-events-none"
        style={{
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -20%)",
          width: "600px",
          height: "500px",
          background:
            "radial-gradient(ellipse 70% 55% at 50% 60%, rgba(201,162,39,0.09) 0%, rgba(201,162,39,0.03) 50%, transparent 72%)",
        }}
        aria-hidden="true"
      />

      <motion.div
        className="relative z-10 max-w-content mx-auto flex flex-col items-center gap-8"
        variants={containerVariants}
        initial="hidden"
        animate="show"
      >
        <motion.div
          variants={itemVariants}
          className="font-mono text-xs tracking-widest uppercase"
          style={{ color: "var(--gold)" }}
        >
          KV-Cache Compression
        </motion.div>

        <motion.h1
          variants={itemVariants}
          className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold leading-tight max-w-3xl"
          style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em", textWrap: "balance" } as React.CSSProperties}
        >
          Most of your context window is never read twice.
        </motion.h1>

        <motion.p
          variants={itemVariants}
          className="text-base sm:text-lg max-w-xl leading-relaxed"
          style={{ color: "var(--paper-muted)" }}
        >
          PageKV groups tokens into fixed-size pages, distills each into one compact summary key,
          then routes each query to only the pages that matter.
        </motion.p>

        {/* Card-stack: float animation when collapsed, disabled for reduced-motion */}
        <motion.div
          variants={itemVariants}
          animate={reduced ? {} : { y: [0, -8, 0] }}
          transition={reduced ? {} : { repeat: Infinity, duration: 5, ease: "easeInOut", delay: 1.2 }}
        >
          <CardStack reduced={reduced} />
        </motion.div>

        <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-3 mt-2">
          <a
            href="#docs"
            className="font-mono text-sm px-6 py-3 min-h-[44px] transition-colors font-medium flex items-center justify-center"
            style={{ backgroundColor: "var(--gold)", color: "var(--void)", borderRadius: "2px" }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.backgroundColor = "var(--gold-bright)")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.backgroundColor = "var(--gold)")}
          >
            View Documentation
          </a>
          <a
            href="https://github.com/VaibhavD2006/PageKV"
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-sm px-6 py-3 min-h-[44px] border transition-colors flex items-center justify-center"
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
        </motion.div>
      </motion.div>
    </section>
  );
}