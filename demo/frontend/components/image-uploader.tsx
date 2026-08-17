"use client"

import Image from "next/image"
import { useRef } from "react"

type Props = {
  file: File | null
  preview: string | null
  error?: string | null
  onFile: (file: File | null) => void
}

export default function ImageUploader({ file, preview, error, onFile }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  return (
    <section className="rounded-3xl border border-line bg-white p-5 shadow-sm">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          onFile(e.dataTransfer.files?.[0] ?? null)
        }}
        onClick={() => inputRef.current?.click()}
        className="cursor-pointer rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50 p-6 text-center transition hover:border-cyan-300 hover:bg-cyan-50/40"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/tiff,image/bmp,image/webp"
          className="hidden"
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
        <p className="text-lg font-semibold text-ink">تصویر رادیوگرافی را اینجا رها کنید</p>
        <p className="mt-1 text-sm text-slate-500">یا برای انتخاب فایل کلیک کنید</p>
        <p className="mt-2 text-xs text-slate-400">فرمت‌های مجاز: JPG، PNG، TIFF، BMP</p>
      </div>
      {file && (
        <div className="mt-4 flex items-center justify-between rounded-2xl border border-line bg-slate-50 px-4 py-3 text-sm">
          <div>
            <div className="font-medium">{file.name}</div>
            <div className="text-slate-500">{Math.round(file.size / 1024)} KB</div>
          </div>
          <button className="text-sm font-medium text-accent" onClick={(e) => { e.stopPropagation(); onFile(null); }}>
            حذف
          </button>
        </div>
      )}
      {preview && (
        <div className="relative mt-4 h-80 w-full overflow-hidden rounded-2xl border border-line bg-white">
          <Image src={preview} alt="پیش‌نمایش" fill className="object-contain" unoptimized />
        </div>
      )}
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
    </section>
  )
}
