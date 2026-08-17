import "@fontsource/vazirmatn/400.css"
import "@fontsource/vazirmatn/500.css"
import "@fontsource/vazirmatn/700.css"
import "./globals.css"
import type { ReactNode } from "react"

export const metadata = {
  title: "سامانه تشخیص شکستگی مچ دست کودکان",
  description: "دموی پژوهشی تشخیص شکستگی با مدل‌های YOLO",
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="fa" dir="rtl">
      <body>{children}</body>
    </html>
  )
}
