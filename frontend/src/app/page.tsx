"use client";

import React, { useEffect, useState } from "react";
import { StatsOverview } from "@/components/dashboard/StatsOverview";
import { SpeciesChart } from "@/components/dashboard/SpeciesChart";
import { ActivityTimeline } from "@/components/dashboard/ActivityTimeline";
import { RecentDetections } from "@/components/dashboard/RecentDetections";
import { Download } from "lucide-react";

export default function Dashboard() {
  const [stats, setStats] = useState({
    total_images: 184,
    animals_detected: 142,
    species_detected: 15,
    tigers_detected: 12,
    auto_accepted: 128,
    pending_review: 14,
    human_reviewed: 32,
    active_cameras: 18,
    automation_rate: 69.5,
  });

  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/analytics/overview")
      .then((res) => res.json())
      .then((data) => {
        if (data?.data) {
          setStats(data.data);
        }
      })
      .catch((err) => console.log("Using baseline stats:", err))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-1">
            SherDrishti Wildlife Intelligence
          </h1>
          <p className="text-xs text-foreground/70">
            Autonomous multi-model camera-trap monitoring & biodiversity intelligence
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs font-mono text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Pipeline Online</span>
          </div>

          <a
            href="http://127.0.0.1:8000/api/analytics/export?format=csv"
            className="px-3 py-1.5 rounded-lg bg-surface-200 hover:bg-surface-300 border border-white/10 text-xs font-medium text-white flex items-center gap-1.5 transition-colors"
          >
            <Download size={14} /> Export CSV
          </a>
        </div>
      </header>

      {/* KPI Stats Overview */}
      <StatsOverview stats={stats} isLoading={isLoading} />

      {/* Live Detections Feed */}
      <RecentDetections />

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ActivityTimeline />
        <SpeciesChart />
      </div>
    </div>
  );
}
