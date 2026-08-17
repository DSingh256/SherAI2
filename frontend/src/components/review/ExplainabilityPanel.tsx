"use client";

import React from "react";
import { CheckCircle2, AlertTriangle, XCircle, Info, ShieldAlert } from "lucide-react";

interface ExplainabilityPanelProps {
  reasoning: string[];
  signals?: Record<string, unknown>;
  isTiger?: boolean;
  isEscalated?: boolean;
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  reasoning = [],
  signals = {},
  isTiger = false,
  isEscalated = false,
}) => {
  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-white/5 pb-2">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground/70 flex items-center gap-1.5">
          <Info size={14} className="text-primary-400" />
          AI Evidence & Reasoning Chain
        </h4>
        {isTiger && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
            🐅 Priority Sighting
          </span>
        )}
        {isEscalated && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">
            <ShieldAlert size={12} /> Conflict Escalated
          </span>
        )}
      </div>

      <div className="space-y-1.5 font-sans text-xs">
        {reasoning.map((item, idx) => {
          const isCheck = item.startsWith("✓");
          const isWarn = item.startsWith("⚠") || item.startsWith("🚨");
          const isCross = item.startsWith("❌");

          return (
            <div
              key={idx}
              className={`flex items-start gap-2 p-2 rounded-lg ${
                isWarn
                  ? "bg-amber-500/10 text-amber-200 border border-amber-500/20"
                  : isCross
                  ? "bg-rose-500/10 text-rose-200 border border-rose-500/20"
                  : "bg-surface-200/50 text-foreground/90 border border-white/5"
              }`}
            >
              {isCheck && <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 shrink-0" />}
              {isWarn && <AlertTriangle size={14} className="text-amber-400 mt-0.5 shrink-0" />}
              {isCross && <XCircle size={14} className="text-rose-400 mt-0.5 shrink-0" />}
              {!isCheck && !isWarn && !isCross && (
                <span className="text-primary-400 mt-0.5 shrink-0">•</span>
              )}
              <span className="leading-relaxed">{item.replace(/^[✓⚠❌🚨]\s*/, "")}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
