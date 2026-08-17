"use client"

import type { PredictResponse } from "@/lib/types"
import { resultImageUrl } from "@/lib/api"
import Image from "next/image"

type Props = {
  original: string | null
  result: PredictResponse | null
}

export default function ResultViewer({ original, result }: Props) {
  return (
    <section className="rounded-3xl border border-line bg-white p-5 shadow-sm">
      {!original ? (
        <div className="flex min-h-[480px] items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50 text-slate-500">
          پس از بارگذاری تصویر، نتیجه اینجا نمایش داده می‌شود.
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <h3 className="mb-2 text-sm font-semibold text-slate-600">تصویر اصلی</h3>
            <div className="relative min-h-[320px] w-full overflow-hidden rounded-2xl border border-line bg-white">
              <Image src={original} alt="تصویر اصلی" fill className="object-contain" unoptimized />
            </div>
          </div>
          <div>
            <h3 className="mb-2 text-sm font-semibold text-slate-600">نتیجه تحلیل</h3>
            {result ? (
              <>
                <div className="relative min-h-[320px] w-full overflow-hidden rounded-2xl border border-line bg-white">
                  <Image src={resultImageUrl(result.annotated_image_url)} alt="نتیجه تحلیل" fill className="object-contain" unoptimized />
                </div>
                <a href={resultImageUrl(result.download_url)} download className="mt-3 inline-flex rounded-xl bg-accent px-4 py-2 text-sm font-medium text-white">
                  دانلود تصویر نتیجه
                </a>
              </>
            ) : (
              <div className="flex min-h-[320px] items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50 text-slate-500">
                در انتظار تحلیل...
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
