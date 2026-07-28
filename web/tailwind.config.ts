import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0B0B0C",
        panel: "#16161A",
        rule: "#2A2A2E",
        paper: "#EDE6D6",
        "paper-muted": "#8A8578",
        gold: "#C9A227",
        "gold-bright": "#E8C468",
        "signal-green": "#6FBF8B",
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        DEFAULT: "2px",
        sm: "2px",
        md: "4px",
        lg: "4px",
        full: "9999px",
      },
      maxWidth: {
        content: "1100px",
      },
    },
  },
  plugins: [],
};

export default config;
