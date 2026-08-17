"use client";

import React, { useState, useEffect, useCallback } from "react";
import { CheckCircle, Search, Filter, RefreshCw, BarChart2, ShieldCheck, Sparkles } from "lucide-react";
import { ReviewQueue } from "@/components/review/ReviewQueue";
import { ReviewCard } from "@/components/review/ReviewCard";


export default function ReviewPage() {
  const [queue, setQueue] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [filterSpecies, setFilterSpecies] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [stats, setStats] = useState({
    verifiedToday: 0,
    agreementRate: 0,
    pending: 0,
  });

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/review/queue")
      .then((res) => res.json())
      .then((data) => {
        if (data?.data?.items) {
          setQueue(data.data.items);
          if (data.data.items.length > 0) {
            setSelectedId(data.data.items[0].image_id);
          }
          setStats((s) => ({ ...s, pending: data.data.items.length }));
        }
      })
      .catch((err) => console.error("Error fetching queue:", err));
      
    fetch("http://127.0.0.1:8000/api/review/stats")
      .then((res) => res.json())
      .then((data) => {
        if (data?.data) {
          setStats((s) => ({
            ...s,
            verifiedToday: data.data.human_reviewed_today || 0,
            agreementRate: data.data.ai_human_agreement_rate || 0,
          }));
        }
      })
      .catch((err) => console.error("Error fetching stats:", err));
  }, []);

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
        await fetch("http://127.0.0.1:8000/api/review/submit", {
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
