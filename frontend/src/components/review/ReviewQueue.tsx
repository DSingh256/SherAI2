"use client";

import React from "react";
import { Clock, ShieldAlert } from "lucide-react";
import { ConfidenceBadge } from "./ConfidenceBadge";

interface ReviewQueueItem {
  id: string;
  camera_id?: string;
  timestamp?: string;
  species?: string;
  confidence?: number;
  decision?: string;
  [key: string]: unknown;
}

interface ReviewQueueProps {
  items: ReviewQueueItem[];
  selectedId: string;
  onSelect: (item: ReviewQueueItem) => void;
  filterSpecies: string;
  setFilterSpecies: (s: string) => void;
}

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  items = [],
  selectedId,
  onSelect,
  filterSpecies,
  setFilterSpecies,
}) => {
  const filtered = filterSpecies
    ? items.filter((i) => i.ai_prediction === filterSpecies)
    : items;

  return (
    <div className="flex flex-col h-full rounded-xl border border-white/10 bg-surface-100/60 overflow-hidden">
      {/* Queue Header */}
      <div className="p-3.5 border-b border-white/10 bg-surface-200/50 flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-white text-sm flex items-center gap-1.5">
            <Clock size={15} className="text-primary-400" />
            Pending Verification
          </h3>
          <p className="text-[11px] text-foreground/60">{filtered.length} camera captures queued</p>
        </div>
        <span className="px-2 py-0.5 rounded-full text-xs font-mono font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
          {filtered.length}
        </span>
      </div>

      {/* Quick Filters */}
      <div className="p-2 border-b border-white/5 bg-surface-100/30 flex gap-1.5 overflow-x-auto text-[11px]">
        <button
          onClick={() => setFilterSpecies("")}
          className={`px-2 py-1 rounded transition-colors whitespace-nowrap ${
            filterSpecies === ""
              ? "bg-primary-600 text-white font-medium"
              : "bg-surface-200 text-foreground/70 hover:text-white"
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilterSpecies("Bengal Tiger")}
          className={`px-2 py-1 rounded transition-colors whitespace-nowrap ${
            filterSpecies === "Bengal Tiger"
              ? "bg-amber-600 text-white font-medium"
              : "bg-surface-200 text-amber-300/80 hover:text-amber-200"
          }`}
        >
          🐅 Tigers Only
        </button>
        <button
          onClick={() => setFilterSpecies("Indian Leopard")}
          className={`px-2 py-1 rounded transition-colors whitespace-nowrap ${
            filterSpecies === "Indian Leopard"
              ? "bg-amber-600 text-white font-medium"
              : "bg-surface-200 text-foreground/70 hover:text-white"
          }`}
        >
          🐆 Leopards
        </button>
      </div>

      {/* Items List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {filtered.length === 0 ? (
          <div className="p-8 text-center text-xs text-foreground/50">
            No items matching active filter
          </div>
        ) : (
          filtered.map((item) => {
            const isSelected = item.image_id === selectedId;
            const isTiger = item.is_tiger || item.ai_prediction === "Bengal Tiger";

            return (
              <div
                key={item.image_id}
                onClick={() => onSelect(item)}
                className={`p-2.5 rounded-lg cursor-pointer transition-all border ${
                  isSelected
                    ? "bg-primary-950/60 border-primary-500/60 shadow-md shadow-primary-950/40"
                    : "bg-surface-200/40 hover:bg-surface-200/80 border-white/5"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-mono text-foreground/50 truncate max-w-[120px]">
                    {item.image_id.slice(0, 8)}...
                  </span>
                  <ConfidenceBadge
                    confidence={item.confidence || 0.6}
                    level={item.confidence_level}
                    size="sm"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <span
                    className={`font-semibold text-xs truncate ${
                      isTiger ? "text-amber-300" : "text-white"
                    }`}
                  >
                    {isTiger && "🐅 "}
                    {item.ai_prediction || "Uncertain Species"}
                  </span>
                  <span className="text-[10px] text-foreground/50 font-mono">
                    {item.camera_id}
                  </span>
                </div>

                {item.decision === "escalated_to_expert" && (
                  <div className="mt-1 flex items-center gap-1 text-[10px] text-rose-300">
                    <ShieldAlert size={11} /> Escalated conflict
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
