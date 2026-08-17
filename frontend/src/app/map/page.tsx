"use client";

import { useState, useEffect } from "react";
// Dynamic import required for Leaflet in Next.js (no SSR)
import dynamic from "next/dynamic";
import { Filter, Layers, Zap } from "lucide-react";
import "leaflet/dist/leaflet.css";

// Dynamically import Map component to avoid SSR window errors
const Map = dynamic(
  () => import("@/components/LiveMap"),
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-surface-100 rounded-xl border border-white/5">
        <div className="w-8 h-8 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin"></div>
      </div>
    )
  }
);

export default function MapPage() {
  return (
    <div className="space-y-6 h-[calc(100vh-8rem)] flex flex-col">
      <header className="flex justify-between items-end mb-2">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Live Map</h1>
          <p className="text-foreground/70">Geospatial view of camera network and recent detections.</p>
        </div>
        <div className="flex gap-4">
          <button className="glass-card px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-surface-200 transition-colors">
            <Layers size={16} />
            <span className="text-sm font-medium">Map Layers</span>
          </button>
          <button className="glass-card px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-surface-200 transition-colors">
            <Filter size={16} />
            <span className="text-sm font-medium">Filters</span>
          </button>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
        {/* Map Area */}
        <div className="lg:col-span-3 glass-card rounded-xl overflow-hidden border border-white/5 relative z-0">
          <Map />
        </div>

        {/* Sidebar Data */}
        <div className="glass-card rounded-xl overflow-hidden flex flex-col border border-white/5">
          <div className="p-4 border-b border-white/5 bg-surface-100/50">
            <h3 className="font-medium text-white flex items-center gap-2">
              <Zap size={16} className="text-accent-500" />
              Live Activity Feed
            </h3>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Feed Items */}
            {[
              { time: "Just now", event: "Tiger Detected", cam: "CAM012", type: "tiger" },
              { time: "15 min ago", event: "Human Activity", cam: "CAM007", type: "threat" },
              { time: "1 hr ago", event: "Wild Boar Sounder", cam: "CAM001", type: "animal" },
              { time: "2 hrs ago", event: "Camera Offline", cam: "CAM022", type: "system" },
              { time: "3 hrs ago", event: "Leopard Sighted", cam: "CAM004", type: "animal" },
              { time: "4 hrs ago", event: "Vehicle Detected", cam: "CAM007", type: "threat" },
            ].map((item, idx) => (
              <div key={idx} className="relative pl-6 pb-4 border-l border-white/10 last:border-0 last:pb-0">
                <div className={`absolute -left-[5px] top-1 w-2.5 h-2.5 rounded-full ${
                  item.type === 'tiger' ? 'bg-accent-500 shadow-[0_0_8px_rgba(245,124,0,0.8)]' :
                  item.type === 'threat' ? 'bg-danger' :
                  item.type === 'system' ? 'bg-warning' : 'bg-primary-500'
                }`}></div>
                <div className="text-xs text-foreground/50 mb-1">{item.time}</div>
                <div className={`font-medium text-sm ${item.type === 'tiger' ? 'text-accent-400' : 'text-white'}`}>
                  {item.event}
                </div>
                <div className="text-xs text-foreground/60">{item.cam}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
