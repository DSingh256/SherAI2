"use client";

import React, { useState, useEffect, useCallback } from "react";
import { CheckCircle, Search, Filter, RefreshCw, BarChart2, ShieldCheck, Sparkles } from "lucide-react";
import { ReviewQueue } from "@/components/review/ReviewQueue";
import { ReviewCard } from "@/components/review/ReviewCard";

const initialMockQueue = [
  {
    image_id: "val-img-001",
    camera_id: "CAM007_Waterhole_Alpha",
    timestamp: "2026-08-17 18:45:22",
    ai_prediction: "Indian Leopard",
    confidence: 0.68,
    confidence_level: "medium",
    decision: "human_review",
    reasoning: [
      "✓ MegaDetector detected animal with 94.2% confidence",
      "✓ SpeciesNet predicts Indian Leopard at 68.0%",
      "⚠ OpenCLIP predicts Jungle Cat at 65.0% (Disagreement)",
      "✓ Human verification recommended before cataloging",
    ],
    image_path: "storage/raw/img_001.jpg",
    is_tiger: false,
  },
  {
    image_id: "val-img-002",
    camera_id: "CAM012_Tiger_Corridor_East",
    timestamp: "2026-08-17 03:15:00",
    ai_prediction: "Bengal Tiger",
    confidence: 0.74,
    confidence_level: "medium",
    decision: "human_review",
    reasoning: [
      "✓ MegaDetector detected animal with 98.5% confidence",
      "✓ SpeciesNet predicts Bengal Tiger at 74.0%",
      "✓ OpenCLIP confirms Bengal Tiger similarity (82.1%)",
      "🐅 TIGER DETECTION — Priority 1 tracking and alert activated",
    ],
    image_path: "storage/raw/img_002.jpg",
    is_tiger: true,
  },
  {
    image_id: "val-img-003",
    camera_id: "CAM022_Buffer_Zone_Village",
    timestamp: "2026-08-17 22:10:05",
    ai_prediction: "Wild Boar",
    confidence: 0.58,
    confidence_level: "low",
    decision: "uncertain",
    reasoning: [
      "✓ MegaDetector detected animal with 78.0% confidence",
      "⚠ SpeciesNet confidence only 58.0%",
      "⚠ Poor night illumination / motion blur",
    ],
    image_path: "storage/raw/img_003.jpg",
    is_tiger: false,
  },
];

export default function ReviewPage() {
  const [queue, setQueue] = useState(initialMockQueue);
  const [selectedId, setSelectedId] = useState(initialMockQueue[0]?.image_id || "");
  const [filterSpecies, setFilterSpecies] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [stats, setStats] = useState({
    verifiedToday: 18,
    agreementRate: 88.5,
    pending: initialMockQueue.length,
  });

  const selectedItem = queue.find((i) => i.image_id === selectedId) || queue[0];

  const handleAction = useCallback(
    async (action: string, humanPrediction?: string, notes?: string) => {
      if (!selectedItem) return;

      const currentIdx = queue.findIndex((i) => i.image_id === selectedItem.image_id);
      const nextQueue = queue.filter((i) => i.image_id !== selectedItem.image_id);

      setQueue(nextQueue);
      setStats((s) => ({
        ...s,
        verifiedToday: s.verifiedToday + 1,
        pending: Math.max(0, s.pending - 1),
      }));

      if (nextQueue.length > 0) {
        const nextItem = nextQueue[currentIdx] || nextQueue[0];
        setSelectedId(nextItem.image_id);
      } else {
        setSelectedId("");
      }

      // Optimistic background sync to backend
      try {
        await fetch("/api/review/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_id: selectedItem.image_id,
            reviewer_id: "Field Biologist",
            action: action.toUpperCase(),
            human_prediction: humanPrediction,
            notes: notes,
          }),
        });
      } catch (err) {
        console.warn("Review sync warning:", err);
      }
    },
    [queue, selectedItem]
  );

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid firing when typing in an input
      if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        handleAction("ACCEPT");
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        handleAction("REJECT");
      } else if (e.key === "c" || e.key === "C") {
        e.preventDefault();
        handleAction("CORRECT");
      } else if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        handleAction("ESCALATE");
      } else if (e.key === "ArrowDown" || e.key === "]") {
        e.preventDefault();
        const curIdx = queue.findIndex((i) => i.image_id === selectedId);
        if (curIdx < queue.length - 1) setSelectedId(queue[curIdx + 1].image_id);
      } else if (e.key === "ArrowUp" || e.key === "[") {
        e.preventDefault();
        const curIdx = queue.findIndex((i) => i.image_id === selectedId);
        if (curIdx > 0) setSelectedId(queue[curIdx - 1].image_id);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleAction, queue, selectedId]);

  return (
    <div className="space-y-4 h-[calc(100vh-7.5rem)] flex flex-col">
      {/* Top Header & Stats Strip */}
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <ShieldCheck className="text-primary-400" size={24} />
            Human-in-the-Loop Review
          </h1>
          <p className="text-xs text-foreground/70">
            Expert verification & ground-truth feedback for uncertain wildlife detections
          </p>
        </div>

        {/* Live Metrics */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="px-3 py-1.5 rounded-lg bg-surface-200 border border-white/5 flex items-center gap-2">
            <span className="text-foreground/60">Verified Today:</span>
            <span className="font-semibold text-emerald-400">{stats.verifiedToday}</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-surface-200 border border-white/5 flex items-center gap-2">
            <span className="text-foreground/60">AI Agreement:</span>
            <span className="font-semibold text-primary-400">{stats.agreementRate}%</span>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center gap-2">
            <span className="text-amber-300/80">Pending Queue:</span>
            <span className="font-bold text-amber-300">{queue.length}</span>
          </div>
        </div>
      </header>

      {/* Main Review Workspace */}
      {queue.length === 0 ? (
        <div className="flex-1 rounded-xl border border-white/10 bg-surface-100/50 flex flex-col items-center justify-center text-center p-8">
          <div className="w-16 h-16 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center mb-4">
            <CheckCircle size={36} />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Queue Fully Verified!</h2>
          <p className="text-xs text-foreground/60 max-w-sm">
            All pending camera-trap captures have been inspected. Autonomous pipeline is monitoring live camera traps.
          </p>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
          {/* Left Column: Queue List */}
          <div className="lg:col-span-1 min-h-0">
            <ReviewQueue
              items={queue}
              selectedId={selectedId}
              onSelect={(item) => setSelectedId(item.image_id)}
              filterSpecies={filterSpecies}
              setFilterSpecies={setFilterSpecies}
            />
          </div>

          {/* Right Column: Interactive Review Card */}
          <div className="lg:col-span-2 rounded-xl border border-white/10 bg-surface-100/60 p-4 overflow-hidden min-h-0">
            <ReviewCard item={selectedItem} onAction={handleAction} />
          </div>
        </div>
      )}
    </div>
  );
}
