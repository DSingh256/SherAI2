"use client";

import React from "react";
import Link from "next/link";
import { ArrowRight, Eye, ShieldAlert, Sparkles } from "lucide-react";

interface RecentDetectionItem {
  image_id: string;
  species: string;
  confidence: number;
  camera_id: string;
  timestamp: string;
  is_tiger?: boolean;
  decision?: string;
  image_path?: string;
}

const mockRecent: RecentDetectionItem[] = [
  {
    image_id: "det-001",
    species: "Bengal Tiger",
    confidence: 0.96,
    camera_id: "CAM007 (Waterhole Alpha)",
    timestamp: "10 mins ago",
    is_tiger: true,
    decision: "auto_accept",
  },
  {
    image_id: "det-002",
    species: "Indian Leopard",
    confidence: 0.88,
    camera_id: "CAM012 (Tiger Trail East)",
    timestamp: "32 mins ago",
    is_tiger: false,
    decision: "auto_accept",
  },
  {
    image_id: "det-003",
    species: "Sambar Deer",
    confidence: 0.94,
    camera_id: "CAM003 (Grassland West)",
    timestamp: "1 hr ago",
    is_tiger: false,
    decision: "auto_accept",
  },
  {
    image_id: "det-004",
    species: "Jungle Cat",
    confidence: 0.65,
    camera_id: "CAM019 (Dense Canopy South)",
    timestamp: "2 hrs ago",
    is_tiger: false,
    decision: "human_review",
  },
];

export const RecentDetections: React.FC<{ items?: RecentDetectionItem[] }> = ({ items = mockRecent }) => {
  const displayItems = items.length > 0 ? items : mockRecent;

  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-white text-base flex items-center gap-2">
            <Sparkles size={16} className="text-primary-400" />
            Live Wildlife Detections
          </h3>
          <p className="text-xs text-foreground/60">Autonomous AI pipeline processing stream</p>
        </div>
        <Link
          href="/review"
          className="text-xs text-primary-400 hover:text-primary-300 flex items-center gap-1 font-medium"
        >
          View queue <ArrowRight size={13} />
        </Link>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {displayItems.slice(0, 4).map((item, idx) => (
          <div
            key={idx}
            className="rounded-lg border border-white/5 bg-surface-200/50 p-3 flex flex-col justify-between hover:bg-surface-200/80 transition-colors"
          >
            <div className="flex items-start justify-between mb-2">
              <span
                className={`text-xs font-semibold px-2 py-0.5 rounded ${
                  item.is_tiger
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    : "bg-surface-300 text-white"
                }`}
              >
                {item.is_tiger && "🐅 "}
                {item.species}
              </span>
              <span className="text-xs font-mono font-medium text-primary-400">
                {(item.confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div className="space-y-1 text-[11px] text-foreground/60">
              <p className="truncate font-mono">{item.camera_id}</p>
              <p className="text-foreground/40">{item.timestamp}</p>
            </div>

            <div className="mt-2 pt-2 border-t border-white/5 flex items-center justify-between text-[11px]">
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] ${
                  item.decision === "auto_accept"
                    ? "text-emerald-400 bg-emerald-500/10"
                    : "text-amber-400 bg-amber-500/10"
                }`}
              >
                {item.decision === "auto_accept" ? "✓ Auto-Accepted" : "⚠ Under Review"}
              </span>
              <Link href="/review" className="text-foreground/50 hover:text-white">
                <Eye size={13} />
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
