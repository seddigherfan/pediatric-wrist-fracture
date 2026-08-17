"use client"

import { useEffect, useMemo, useState } from "react"
import Disclaimer from "@/components/disclaimer"
import ImageUploader from "@/components/image-uploader"
import ModelSelector from "@/components/model-selector"
import ResultViewer from "@/components/result-viewer"
import DetectionSummary from "@/components/detection-summary"
import { fetchModels, predictImage } from "@/lib/api"
import type { ModelInfo, PredictResponse } from "@/lib/types"

export default function Page() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selected, setSelected] = useState("yolov8")
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0.25)

  useEffect(() => {
    fetchModels().then(setModels).catch(() => setError("ارتباط با backend برقرار نشد."))
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const selectedModel = useMemo(() => models.find((m) => m.id === selected), [models, selected])

  async function onSubmit() {
    if (!file) return setError("ابتدا یک تصویر معتبر انتخاب کنید.")
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await predictImage({ file, modelId: selected, confidence })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : "تحلیل تصویر ناموفق بود.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_right,_rgba(56,189,248,0.12),_transparent_30%),radial-gradient(circle_at_left,_rgba(20,93,160,0.08),_transparent_35%)] px-4 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="rounded-3xl border border-line bg-white p-6 shadow-sm">
          <div className="inline-flex rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-accent">نسخه پژوهشی</div>
          <h1 className="mt-4 text-3xl font-bold text-ink md:text-4xl">سامانه تشخیص شکستگی مچ دست کودکان</h1>
          <p className="mt-2 text-slate-600">تحلیل تصاویر رادیوگرافی با مدل‌های تشخیص شیء</p>
          <div className="mt-4">
            <Disclaimer />
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-[420px_1fr]">
          <aside className="space-y-4">
            <ImageUploader file={file} preview={preview} error={error} onFile={(next) => { setFile(next); setError(null); setResult(null); }} />
            <ModelSelector models={models} value={selected} onChange={setSelected} />
            <section className="rounded-3xl border border-line bg-white p-5 shadow-sm">
              <label className="mb-2 block text-sm font-semibold">آستانه اطمینان: {Math.round(confidence * 100)}٪</label>
              <input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} className="w-full" />
              <div className="mt-4 flex gap-3">
                <button disabled={loading || !file || !selectedModel?.checkpoint_available} onClick={onSubmit} className="flex-1 rounded-2xl bg-accent px-4 py-3 font-semibold text-white disabled:opacity-50">
                  {loading ? "در حال تحلیل تصویر..." : "تحلیل تصویر"}
                </button>
                <button onClick={() => { setFile(null); setResult(null); setError(null); }} className="rounded-2xl border border-line px-4 py-3 font-semibold text-slate-700">
                  پاک‌کردن
                </button>
              </div>
              {!selectedModel?.checkpoint_available && <p className="mt-3 text-sm text-amber-700">checkpoint این مدل در دسترس نیست.</p>}
              {loading && <p className="mt-3 text-sm text-slate-500">در حال تحلیل تصویر...</p>}
            </section>
          </aside>

          <section className="space-y-4">
            {result && <DetectionSummary result={result} />}
            <ResultViewer original={preview} result={result} />
          </section>
        </div>
      </div>
    </main>
  )
}
