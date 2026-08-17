import type { Config } from "tailwindcss"

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Vazirmatn", "sans-serif"],
      },
      colors: {
        ink: "#10233d",
        mist: "#f7f9fc",
        line: "#d8e2ef",
      },
    },
  },
  plugins: [],
} satisfies Config

