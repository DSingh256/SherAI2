"use client";

import React from "react";
import { Sparkles, Shield } from "lucide-react";

interface SpeciesItem {
  species: string;
  count: number;
  percentage: number;
  avg_confidence: number;
  is_tiger?: boolean;
}

interface SpeciesDistributionProps {
  distribution?: SpeciesItem[];
}

const defaultDist: SpeciesItem[] = [
  { species: "Spotted Deer (Chital)", count: 68, percentage: 31.2, avg_confidence: 0.94 },
  { species: "Sambar Deer", count: 42, percentage: 19.3, avg_confidence: 0.91 },
  { species: "Wild Boar", count: 34, percentage: 15.6, avg_confidence: 0.88 },
  { species: "Bengal Tiger", count: 18, percentage: 8.3, avg_confidence: 0.96, is_tiger: true },
  { species: "Indian Leopard", count: 14, percentage: 6.4, avg_confidence: 0.89 },
  { species: "Common Langur", count: 20, percentage: 9.2, avg_confidence: 0.92 },
  { species: "Asian Elephant", count: 9, percentage: 4.1, avg_confidence: 0.95 },
  { species: "Sloth Bear", count: 7, percentage: 3.2, avg_confidence: 0.87 },
  { species: "Golden Jackal", count: 6, percentage: 2.7, avg_confidence: 0.84 },
];

export const SpeciesDistribution: React.FC<SpeciesDistributionProps> = ({ distribution = defaultDist }) => {
  const items = distribution.length > 0 ? distribution : defaultDist;

  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-5 space-y-4">
      <div>
        <h3 className="font-semibold text-white text-base flex items-center gap-2">
          <Sparkles size={16} className="text-primary-400" />
          Species Population Census & Confidence
        </h3>
        <p className="text-xs text-foreground/60">Verified wildlife observations ranking</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-white/10 text-foreground/50 uppercase tracking-wider font-mono">
              <th className="pb-2">Species</th>
              <th className="pb-2 text-right">Detections</th>
              <th className="pb-2 text-right">Share</th>
              <th className="pb-2 text-right">Avg Confidence</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5 font-mono">
            {items.map((row, idx) => (
              <tr key={idx} className="hover:bg-surface-200/50 transition-colors">
                <td className="py-2.5 font-sans font-medium text-white flex items-center gap-2">
                  {row.is_tiger && <span>🐅</span>}
                  <span className={row.is_tiger ? "text-amber-300 font-semibold" : ""}>
                    {row.species}
                  </span>
                </td>
                <td className="py-2.5 text-right font-bold text-foreground/90">{row.count}</td>
                <td className="py-2.5 text-right text-foreground/60">{row.percentage}%</td>
                <td className="py-2.5 text-right">
                  <span className="px-2 py-0.5 rounded text-[11px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {(row.avg_confidence * 100).toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
