"use client";

import React, { useEffect, useState } from "react";
import { TemporalChart } from "@/components/analytics/TemporalChart";
import { SpeciesDistribution } from "@/components/analytics/SpeciesDistribution";
import { CameraHotspots } from "@/components/analytics/CameraHotspots";
import { Download, FileJson, TrendingUp } from "lucide-react";

export default function AnalyticsPage() {
  const [speciesData, setSpeciesData] = useState([]);
  const [temporalData, setTemporalData] = useState<Record<string, unknown> | null>(null);
  const [hotspotData, setHotspotData] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/analytics/species")
      .then((r) => r.json())
      .then((d) => setSpeciesData(d?.data?.distribution || []))
      .catch((e) => console.log("Baseline species:", e));

    fetch("http://127.0.0.1:8000/api/analytics/temporal")
      .then((r) => r.json())
      .then((d) => setTemporalData(d?.data || null))
      .catch((e) => console.log("Baseline temporal:", e));

    fetch("http://127.0.0.1:8000/api/analytics/cameras")
      .then((r) => r.json())
      .then((d) => setHotspotData(d?.data?.hotspots || []))
      .catch((e) => console.log("Baseline cameras:", e));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-1 flex items-center gap-2">
            <TrendingUp className="text-primary-400" size={28} />
            Wildlife & Conservation Analytics
          </h1>
          <p className="text-xs text-foreground/70">
            Circadian activity cycles, population census, and wildlife corridor density analysis
          </p>
        </div>

        {/* Report Export Buttons */}
        <div className="flex items-center gap-2">
          <a
            href="http://127.0.0.1:8000/api/analytics/export?format=csv"
            className="px-3.5 py-2 rounded-lg bg-surface-200 hover:bg-surface-300 border border-white/10 text-xs font-semibold text-white flex items-center gap-2 transition-colors"
          >
            <Download size={14} className="text-emerald-400" /> Export CSV
          </a>
          <a
            href="http://127.0.0.1:8000/api/analytics/export?format=json"
            className="px-3.5 py-2 rounded-lg bg-surface-200 hover:bg-surface-300 border border-white/10 text-xs font-semibold text-white flex items-center gap-2 transition-colors"
          >
            <FileJson size={14} className="text-primary-400" /> Export JSON
          </a>
        </div>
      </header>

      {/* Circadian Activity Curve */}
      <TemporalChart
        hourly={temporalData?.hourly}
        dayNightSplit={temporalData?.day_night_split}
      />

      {/* Grid: Species Population Table + Camera Hotspots */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <SpeciesDistribution distribution={speciesData} />
        <CameraHotspots hotspots={hotspotData} />
      </div>
    </div>
  );
}
