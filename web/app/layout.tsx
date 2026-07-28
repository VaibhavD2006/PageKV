import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PageKV — KV-Cache Compression for Long-Context LLMs",
  description:
    "PageKV compresses transformer KV-cache memory by grouping tokens into pages and routing attention only to the pages that matter. Drop-in for HuggingFace models.",
  openGraph: {
    title: "PageKV",
    description: "KV-Cache Compression for Long-Context LLMs",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
