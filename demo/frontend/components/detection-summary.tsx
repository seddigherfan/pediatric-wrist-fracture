"use client"

import type { PredictResponse } from "@/lib/types"

export default function DetectionSummary({ result }: { result: PredictResponse }) {
  const detected = result.fracture_detected ? "شکستگی مشاهده شد" : "شکستگی مشاهده نشد"
  return (
    <section className="rounded-3xl border border-line bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold">خلاصه تحلیل</h2>
      <div className={`mt-3 rounded-2xl p-4 text-sm ${result.fracture_detected ? "bg-rose-50 text-rose-900" : "bg-emerald-50 text-emerald-900"}`}>{detected}</div>
      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <Stat label="تعداد نواحی" value={String(result.num_detections)} />
        <Stat label="بیشترین اطمینان" value={`${Math.round(result.maximum_confidence * 100)}٪`} />
        <Stat label="مدل" value={result.model_display_name} />
        <Stat label="زمان inference" value={`${result.inference_time_ms.toFixed(1)} ms`} />
        <Stat label="زمان کل" value={`${result.total_processing_time_ms.toFixed(1)} ms`} />
        <Stat label="ابعاد" value={`${result.original_width}×${result.original_height}`} />
      </dl>
    </section>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-line bg-slate-50 p-3">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-1 font-medium">{value}</dd>
    </div>
  )
}
