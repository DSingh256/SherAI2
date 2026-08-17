"use client";

import React from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

interface SpeciesChartProps {
  data?: Array<{ species: string; count: number; percentage?: number }>;
}

const defaultData = [
  { species: "Spotted Deer", count: 52 },
  { species: "Sambar Deer", count: 38 },
  { species: "Wild Boar", count: 26 },
  { species: "Bengal Tiger", count: 12 },
  { species: "Indian Leopard", count: 9 },
  { species: "Common Langur", count: 14 },
  { species: "Asian Elephant", count: 6 },
];

export const SpeciesChart: React.FC<SpeciesChartProps> = ({ data = defaultData }) => {
  const chartData = data.length > 0 ? data.slice(0, 7) : defaultData;

  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-5 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-semibold text-white text-base">Species Distribution</h3>
          <p className="text-xs text-foreground/60">Population census across camera network</p>
        </div>
      </div>

      <div className="flex-1 min-h-[240px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              dataKey="species"
              type="category"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: "rgba(255,255,255,0.8)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={110}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-surface-100, #141f17)",
                borderColor: "rgba(255,255,255,0.1)",
                borderRadius: "8px",
                color: "#fff",
                fontSize: "12px",
              }}
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
            />
            <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
