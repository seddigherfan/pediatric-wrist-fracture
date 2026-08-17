"use client"

import type { ModelInfo } from "@/lib/types"

type Props = { models: ModelInfo[]; value: string; onChange: (value: string) => void }

export default function ModelSelector({ models, value, onChange }: Props) {
  return (
    <section className="rounded-3xl border border-line bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold">انتخاب مدل</h2>
      <div className="mt-4 space-y-3">
        {models.map((model) => (
          <button
            key={model.id}
            disabled={!model.checkpoint_available}
            onClick={() => onChange(model.id)}
            className={`w-full rounded-2xl border p-4 text-right transition ${
              value === model.id ? "border-accent bg-blue-50" : "border-line bg-slate-50"
            } ${!model.checkpoint_available ? "cursor-not-allowed opacity-50" : "hover:border-accent/50"}`}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="font-semibold">{model.display_name}</div>
                <div className="text-sm text-slate-500">{model.english_name}</div>
                <div className="mt-1 text-xs text-slate-500">{model.description}</div>
              </div>
              <span className="rounded-full px-3 py-1 text-xs font-medium">
                {model.checkpoint_available ? "آماده" : "در دسترس نیست"}
              </span>
            </div>
          </button>
        ))}
      </div>
    </section>
  )
}
