"use client";

import React, { useState } from "react";
import { 
  Check, X, Edit3, ShieldAlert, 
  Layers, Eye, ZoomIn 
} from "lucide-react";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { ExplainabilityPanel } from "./ExplainabilityPanel";

interface ReviewItem {
  id: string;
  camera_id?: string;
  timestamp?: string;
  species?: string;
  confidence?: number;
  decision?: string;
  megadetector_confidence?: number;
  speciesnet_confidence?: number;
  openclip_similarity?: number;
  segmentation_path?: string;
  image_path?: string;
  reasoning?: string[];
  signals?: Record<string, unknown>;
  [key: string]: unknown;
}

interface ReviewCardProps {
  item: ReviewItem;
  onAction: (action: string, humanPrediction?: string, notes?: string) => void;
}

const SPECIES_OPTIONS = [
  "Bengal Tiger",
  "Indian Leopard",
  "Asian Elephant",
  "Sambar Deer",
  "Spotted Deer (Chital)",
  "Wild Boar",
  "Sloth Bear",
  "Indian Gaur",
  "Nilgai",
  "Golden Jackal",
  "Dhole",
  "Striped Hyena",
  "Jungle Cat",
  "Rhesus Macaque",
  "Common Langur",
  "Human",
  "Vehicle",
  "False Alarm / Empty",
];

export const ReviewCard: React.FC<ReviewCardProps> = ({ item, onAction }) => {
  const [selectedSpecies, setSelectedSpecies] = useState(item?.ai_prediction || "Bengal Tiger");
  const [notes, setNotes] = useState("");
  const [showBBoxes, setShowBBoxes] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);

  if (!item) {
    return (
      <div className="flex-1 flex items-center justify-center p-12 text-center text-foreground/50">
        Select an image from the queue to start verification
      </div>
    );
  }

  const primaryConfidence = item.confidence || 0.0;
  const isTiger = item.is_tiger || item.ai_prediction === "Bengal Tiger";

  return (
    <div className="flex flex-col h-full overflow-y-auto space-y-4">
      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-xl font-bold text-white tracking-tight">
              AI Prediction: <span className="text-primary-400">{item.ai_prediction || "Animal"}</span>
            </h2>
            <ConfidenceBadge confidence={primaryConfidence} level={item.confidence_level} size="md" />
          </div>
          <p className="text-xs text-foreground/60">
            Camera: <span className="text-foreground/90 font-mono">{item.camera_id}</span> • Timestamp:{" "}
            <span className="text-foreground/90">{item.timestamp || "Live Capture"}</span>
          </p>
        </div>

        {/* View Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowBBoxes(!showBBoxes)}
            className={`px-2.5 py-1 text-xs rounded border transition-colors flex items-center gap-1 ${
              showBBoxes
                ? "bg-primary-500/20 text-primary-300 border-primary-500/40"
                : "bg-surface-200 text-foreground/60 border-white/5"
            }`}
            title="Toggle MegaDetector Bounding Box"
          >
            <Layers size={13} /> Boxes
          </button>
          <button
            onClick={() => setShowMask(!showMask)}
            className={`px-2.5 py-1 text-xs rounded border transition-colors flex items-center gap-1 ${
              showMask
                ? "bg-accent-500/20 text-accent-300 border-accent-500/40"
                : "bg-surface-200 text-foreground/60 border-white/5"
            }`}
            title="Toggle SAM2 Segmentation Mask"
          >
            <Eye size={13} /> Mask
          </button>
          <button
            onClick={() => setZoomLevel((z) => (z >= 2 ? 1 : z + 0.5))}
            className="p-1.5 rounded bg-surface-200 hover:bg-surface-300 text-foreground/70 border border-white/5"
            title="Zoom In"
          >
            <ZoomIn size={14} />
          </button>
        </div>
      </div>

      {/* Side-by-Side Visuals */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 min-h-[260px] max-h-[340px]">
        {/* Frame 1: Original Capture with BBoxes */}
        <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/60 flex items-center justify-center group">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={item.image_path ? `/${item.image_path}` : "https://images.unsplash.com/photo-1544641957-3f360c497424?w=800&q=80"}
            alt="Original capture"
            style={{ transform: `scale(${zoomLevel})` }}
            className="w-full h-full object-contain transition-transform duration-200"
          />
          {showBBoxes && (
            <div className="absolute inset-[15%] border-2 border-emerald-400/90 rounded bg-emerald-400/10 pointer-events-none flex items-start p-1">
              <span className="text-[10px] font-mono bg-emerald-950/90 text-emerald-300 px-1 py-0.5 rounded border border-emerald-400/40">
                MegaDetector: Animal ({(primaryConfidence * 100).toFixed(0)}%)
              </span>
            </div>
          )}
          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[11px] font-mono text-white/80 border border-white/10">
            Raw Camera Frame
          </div>
        </div>

        {/* Frame 2: SAM2 Segmented Wildlife Crop */}
        <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/60 flex items-center justify-center">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={item.image_path ? `/${item.image_path}` : "https://images.unsplash.com/photo-1579753767215-6217435f3032?w=800&q=80"}
            alt="Segmented crop"
            style={{ transform: `scale(${zoomLevel})` }}
            className="w-full h-full object-contain transition-transform duration-200"
          />
          {showMask && (
            <div className="absolute inset-[20%] border-2 border-amber-400/80 rounded-full bg-amber-400/15 pointer-events-none flex items-end justify-center p-1">
              <span className="text-[10px] font-mono bg-amber-950/90 text-amber-300 px-1 py-0.5 rounded border border-amber-400/40">
                SAM2 Mask: 92% Quality
              </span>
            </div>
          )}
          <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 text-[11px] font-mono text-amber-300 border border-amber-500/20">
            SAM2 Isolated Crop (Re-ID Ready)
          </div>
        </div>
      </div>

      {/* Explainability Reasoning */}
      <ExplainabilityPanel
        reasoning={item.reasoning || []}
        signals={item.signals || {}}
        isTiger={isTiger}
        isEscalated={item.decision === "escalated_to_expert"}
      />

      {/* Human Actions & Correction Controls */}
      <div className="rounded-xl border border-white/10 bg-surface-100/90 p-4 space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground/70">
          Human Verification & Decision
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-[11px] text-foreground/60 mb-1 font-medium">
              Confirm or Correct Species:
            </label>
            <select
              value={selectedSpecies}
              onChange={(e) => setSelectedSpecies(e.target.value)}
              className="w-full p-2.5 rounded-lg text-sm bg-surface-200 border border-white/10 text-white focus:ring-2 focus:ring-primary-500"
            >
              {SPECIES_OPTIONS.map((sp) => (
                <option key={sp} value={sp}>
                  {sp === item.ai_prediction ? `✓ ${sp} (AI Match)` : sp}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[11px] text-foreground/60 mb-1 font-medium">
              Reviewer Notes (Optional):
            </label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g., Juvenile male tiger with distinct left flank stripes..."
              className="w-full p-2.5 rounded-lg text-sm bg-surface-200 border border-white/10 text-white placeholder-foreground/40"
            />
          </div>
        </div>

        {/* Buttons Action Bar */}
        <div className="flex flex-wrap gap-2 pt-2 border-t border-white/5">
          <button
            onClick={() => onAction("ACCEPT", selectedSpecies, notes)}
            className="flex-1 min-w-[130px] bg-emerald-600 hover:bg-emerald-500 text-white py-2.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
          >
            <Check size={15} /> Accept (A)
          </button>

          <button
            onClick={() => onAction("CORRECT", selectedSpecies, notes)}
            className="flex-1 min-w-[130px] bg-primary-600 hover:bg-primary-500 text-white py-2.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
          >
            <Edit3 size={15} /> Correct (C)
          </button>

          <button
            onClick={() => onAction("REJECT", "None", notes)}
            className="px-4 py-2.5 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
          >
            <X size={15} /> Reject (R)
          </button>

          <button
            onClick={() => onAction("ESCALATE", selectedSpecies, notes)}
            className="px-4 py-2.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
          >
            <ShieldAlert size={15} /> Escalate (E)
          </button>
        </div>
      </div>
    </div>
  );
};
