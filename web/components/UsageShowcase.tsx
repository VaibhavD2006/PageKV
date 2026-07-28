"use client";
import { useState } from "react";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard?.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <button
      onClick={copy}
      aria-label="Copy to clipboard"
      className="font-mono text-xs px-2 py-1 transition-colors cursor-pointer min-h-[36px]"
      style={{
        border: "1px solid var(--rule)",
        color: copied ? "var(--signal-green)" : "var(--paper-muted)",
        borderRadius: "2px",
        backgroundColor: "transparent",
      }}
    >
      {copied ? "✓ copied" : "copy"}
    </button>
  );
}

const blocks = [
  {
    lang: "bash",
    label: "Install",
    code: `pip install pagekv`,
  },
  {
    lang: "python",
    label: "Patch your model",
    code: `from transformers import AutoModelForCausalLM
from pagekv import patch_model

model = AutoModelForCausalLM.from_pretrained("MODEL_NAME")
patch_model(model, page_size=128, top_k_pages=4)

# model now uses paged attention — generation API unchanged
outputs = model.generate(input_ids, max_new_tokens=256)`,
  },
];

export default function UsageShowcase() {
  return (
    <section id="usage" className="px-6 py-24">
      <div className="mx-auto max-w-content">
        <div className="mb-10">
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--paper-muted)" }}>
            Usage
          </div>
          <h2
            className="font-display text-3xl sm:text-4xl font-bold leading-tight"
            style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em" } as React.CSSProperties}
          >
            Two lines to patch any HuggingFace model.
          </h2>
        </div>

        <div className="space-y-4">
          {blocks.map((block) => (
            <div
              key={block.label}
              style={{
                borderRadius: "2px",
                border: "1px solid var(--rule)",
                overflow: "hidden",
              }}
            >
              {/* terminal title bar */}
              <div
                className="flex items-center justify-between px-4 py-2"
                style={{
                  backgroundColor: "var(--panel)",
                  borderBottom: "1px solid var(--gold)",
                }}
              >
                <span className="font-mono text-xs" style={{ color: "var(--paper-muted)" }}>
                  {block.lang === "bash" ? "$ " : ">>> "}
                  {block.label}
                </span>
                <CopyButton text={block.code} />
              </div>
              <pre
                className="px-5 py-4 overflow-x-auto font-mono text-sm leading-relaxed"
                style={{ backgroundColor: "var(--panel)", color: "var(--paper)", margin: 0 }}
              >
                <code>{block.code}</code>
              </pre>
            </div>
          ))}
        </div>

        <div className="mt-6 font-mono text-xs" style={{ color: "var(--paper-muted)" }}>
          The generation API is unchanged.{" "}
          <code style={{ color: "var(--paper)" }}>patch_model</code> replaces the attention forward pass in-place.
        </div>
      </div>
    </section>
  );
}
