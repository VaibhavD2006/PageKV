"use client";

/* Mini card-stack motif reused as section divider throughout the page */
export default function SectionDivider() {
  return (
    <div className="flex items-center gap-4 py-2">
      <hr className="divider-gold flex-1" />
      <svg width="20" height="16" viewBox="0 0 20 16" fill="none" aria-hidden="true">
        <rect x="2" y="6" width="16" height="10" rx="1" stroke="#2A2A2E" strokeWidth="1" fill="#16161A" />
        <rect x="4" y="3" width="12" height="10" rx="1" stroke="#2A2A2E" strokeWidth="1" fill="#16161A" />
        <rect x="6" y="0" width="8" height="10" rx="1" stroke="#C9A227" strokeWidth="1" fill="#16161A" />
      </svg>
      <hr className="divider-gold flex-1" style={{ transform: "scaleX(-1)" }} />
    </div>
  );
}
