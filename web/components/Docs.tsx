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

function CodeBlock({ lang, label, code }: { lang: string; label: string; code: string }) {
  return (
    <div style={{ borderRadius: "2px", border: "1px solid var(--rule)", overflow: "hidden" }}>
      <div
        className="flex items-center justify-between px-4 py-2"
        style={{ backgroundColor: "var(--panel)", borderBottom: "1px solid var(--gold)" }}
      >
        <span className="font-mono text-xs" style={{ color: "var(--paper-muted)" }}>
          {lang === "bash" ? "$ " : ">>> "}
          {label}
        </span>
        <CopyButton text={code} />
      </div>
      <pre
        className="px-5 py-4 overflow-x-auto font-mono text-sm leading-relaxed"
        style={{ backgroundColor: "var(--panel)", color: "var(--paper)", margin: 0 }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

const sections = [
  {
    id: "docs-core",
    title: "Standalone semantic search",
    description:
      "pagekv.Index gives you a fast two-stage retrieval index with no external dependencies. Bring your own embeddings — any model works.",
    install: null,
    blocks: [
      {
        lang: "python",
        label: "index.py",
        code: `import numpy as np
from pagekv import Index, SearchResult

# Build from pre-computed embeddings (any [N, D] float32 array)
embeddings = np.load("embeddings.npy")   # shape [N, D]
texts = open("chunks.txt").read().splitlines()

index = Index.from_embeddings(
    embeddings, texts,
    page_size=128,    # chunks per page
    top_k_pages=4,    # coarse-pass page budget
)

# Search
query_emb = embed_model("What is attention?")   # [D] float32
results: list[SearchResult] = index.search(query_emb, top_k=5)

for r in results:
    print(f"[{r.score:.3f}] {r.text[:80]}")

# Persist — no pickle, safe numpy .npz
index.save("my_index.npz")
loaded = Index.load("my_index.npz")`,
      },
    ],
  },
  {
    id: "docs-embed",
    title: "With sentence-transformers",
    description:
      "Go from raw text to a searchable index in one step using any sentence-transformers model — included with pagekv.",
    install: null,
    blocks: [
      {
        lang: "python",
        label: "embed_index.py",
        code: `from pagekv import Index
from pagekv.embed import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

texts = [
    "The capital of France is Paris.",
    "PyTorch is a deep learning framework.",
    "Transformers use attention mechanisms.",
]

# embed() returns [N, D] float32, L2-normalised
index = Index.from_embeddings(embedder.embed(texts), texts, page_size=2)

query = embedder.embed_one("What is the capital of France?")
results = index.search(query, top_k=1)
print(results[0].text)  # → "The capital of France is Paris."`,
      },
    ],
  },
  {
    id: "docs-langchain",
    title: "LangChain retriever",
    description:
      "Drop PageKV into any LangChain RAG pipeline. PageKVRetriever wraps pagekv.Index and returns standard LangChain Document objects.",
    install: null,
    blocks: [
      {
        lang: "python",
        label: "langchain_rag.py",
        code: `from langchain_openai import OpenAIEmbeddings
from pagekv.integrations.langchain_retriever import PageKVRetriever

embed_fn = OpenAIEmbeddings().embed_documents   # list[str] -> list[list[float]]

retriever = PageKVRetriever.from_texts(
    texts=my_chunks,
    embedding_function=embed_fn,
    page_size=128,
    top_k_pages=4,
    top_k=10,
)

# Works with any LangChain chain
docs = retriever.get_relevant_documents("explain attention mechanisms")

# Or wrap an existing pagekv.Index
from pagekv import Index
index = Index.load("my_index.npz")
retriever = PageKVRetriever.from_index(index, embed_fn, top_k=5)`,
      },
    ],
  },
  {
    id: "docs-llamaindex",
    title: "LlamaIndex retriever",
    description:
      "Use PageKVNodeRetriever inside any LlamaIndex query engine. It returns NodeWithScore objects compatible with the full LlamaIndex ecosystem.",
    install: null,
    blocks: [
      {
        lang: "python",
        label: "llamaindex_rag.py",
        code: `from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from pagekv.integrations.llamaindex_retriever import PageKVNodeRetriever

docs = SimpleDirectoryReader("./data").load_data()
nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(docs)

retriever = PageKVNodeRetriever.from_nodes(
    nodes=nodes,
    embed_model=embed_model,   # any LlamaIndex embed model
    page_size=128,
    top_k_pages=4,
    top_k=10,
)

results = retriever.retrieve("explain self-attention")

# Or wrap an existing pagekv.Index
from pagekv import Index
index = Index.load("my_index.npz")
retriever = PageKVNodeRetriever.from_index(index, embed_model, top_k=5)`,
      },
    ],
  },
];

export default function Docs() {
  return (
    <section id="docs" className="px-6 py-24">
      <div className="mx-auto max-w-content">
        <div className="mb-12">
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--paper-muted)" }}>
            Integration Guide
          </div>
          <h2
            className="font-display text-3xl sm:text-4xl font-bold leading-tight mb-4"
            style={{ fontOpticalSizing: "auto", letterSpacing: "-0.02em" } as React.CSSProperties}
          >
            Integrate pagekv.Index into your project.
          </h2>
          <p className="text-base max-w-2xl mb-6" style={{ color: "var(--paper-muted)" }}>
            pagekv.Index is a standalone semantic search engine — no server, no cloud, no Dockerfile. Runs in your
            Python process alongside your existing RAG stack.
          </p>
          <div className="inline-block">
            <CodeBlock lang="bash" label="install" code="pip install pagekv" />
          </div>
        </div>

        <div className="space-y-16">
          {sections.map((section) => (
            <div key={section.id} id={section.id}>
              <div className="mb-5">
                <h3
                  className="font-display text-xl font-semibold mb-2"
                  style={{ letterSpacing: "-0.01em" } as React.CSSProperties}
                >
                  {section.title}
                </h3>
                <p className="text-sm" style={{ color: "var(--paper-muted)" }}>
                  {section.description}
                </p>
              </div>
              <div className="space-y-3">
                {section.blocks.map((b) => (
                  <CodeBlock key={b.label} lang={b.lang} label={b.label} code={b.code} />
                ))}
              </div>
            </div>
          ))}
        </div>

        <div
          className="mt-16 p-6"
          style={{ border: "1px solid var(--rule)", borderRadius: "2px", backgroundColor: "var(--panel)" }}
        >
          <div className="font-mono text-xs tracking-widest uppercase mb-3" style={{ color: "var(--gold)" }}>
            SearchResult fields
          </div>
          <div className="font-mono text-sm space-y-1.5" style={{ color: "var(--paper)" }}>
            {[
              { name: "text", type: "str", note: "original chunk text" },
              { name: "score", type: "float", note: "dot-product similarity (higher = more relevant)" },
              { name: "chunk_id", type: "int", note: "position of this chunk in the original texts list" },
              { name: "page_id", type: "int", note: "which page this chunk belongs to" },
            ].map((f) => (
              <div key={f.name}>
                <span style={{ color: "var(--gold)" }}>{f.name}</span>
                <span style={{ color: "var(--paper-muted)" }}>: {f.type}</span>
                <span className="ml-4 text-xs" style={{ color: "var(--paper-muted)" }}>
                  {f.note}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
