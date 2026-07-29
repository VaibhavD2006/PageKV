"use client";
import { useEffect, useState } from "react";

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-300"
      style={{
        backgroundColor: scrolled ? "rgba(22,22,26,0.85)" : "transparent",
        backdropFilter: scrolled ? "blur(12px)" : "none",
        WebkitBackdropFilter: scrolled ? "blur(12px)" : "none",
        borderBottom: scrolled ? "1px solid var(--rule)" : "none",
      }}
    >
      <div className="mx-auto max-w-content px-6 flex items-center justify-between h-14">
        <span
          className="font-display text-gold font-semibold text-lg tracking-tight"
          style={{ fontOpticalSizing: "auto" } as React.CSSProperties}
        >
          PageKV
        </span>
        <div className="flex items-center gap-6">
          <a href="#how-it-works" className="text-paper-muted hover:text-paper text-sm transition-colors hidden sm:block">
            Docs
          </a>
          <a href="#usage" className="text-paper-muted hover:text-paper text-sm transition-colors hidden sm:block">
            Usage
          </a>
          <a
            href="https://github.com/VaibhavD2006/PageKV"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-sm border border-gold text-gold hover:bg-gold hover:text-void px-3 py-1.5 rounded-full transition-colors min-h-[44px] sm:min-h-[36px]"
          >
            ★ Star on GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}