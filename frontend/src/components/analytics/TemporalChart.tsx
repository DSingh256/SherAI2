"use client";

import React from "react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { Moon, Sun, Clock } from "lucide-react";

interface TemporalChartProps {
  hourly?: Array<{ hour: number; label: string; count: number }>;
  dayNightSplit?: {
    day_activity: number;
    night_activity: number;
    night_percentage: number;
  };
}

const defaultHourly = [
  { hour: 0, label: "00:00", count: 18 },
  { hour: 2, label: "02:00", count: 24 },
  { hour: 4, label: "04:00", count: 29 },
  { hour: 6, label: "06:00", count: 35 },
  { hour: 8, label: "08:00", count: 14 },
  { hour: 10, label: "10:00", count: 8 },
  { hour: 12, label: "12:00", count: 6 },
  { hour: 14, label: "14:00", count: 7 },
  { hour: 16, label: "16:00", count: 15 },
  { hour: 18, label: "18:00", count: 38 },
  { hour: 20, label: "20:00", count: 31 },
  { hour: 22, label: "22:00", count: 22 },
];

export const TemporalChart: React.FC<TemporalChartProps> = ({
  hourly = defaultHourly,
  dayNightSplit = { day_activity: 85, night_activity: 162, night_percentage: 65.6 },
}) => {
  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="font-semibold text-white text-base flex items-center gap-2">
            <Clock size={16} className="text-primary-400" />
            Circadian Wildlife Activity Pattern
          </h3>
          <p className="text-xs text-foreground/60">24-hour activity distribution across cameras</p>
        </div>

        {/* Day / Night Breakdown Pill */}
        <div className="flex items-center gap-2 text-xs font-mono bg-surface-200/80 px-3 py-1.5 rounded-lg border border-white/5">
          <span className="flex items-center gap-1 text-amber-300">
            <Sun size={13} /> {dayNightSplit.day_activity} Day
          </span>
          <span className="text-foreground/30">•</span>
          <span className="flex items-center gap-1 text-indigo-300">
            <Moon size={13} /> {dayNightSplit.night_activity} Night ({dayNightSplit.night_percentage}%)
          </span>
        </div>
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={hourly} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="label"
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="rgba(255,255,255,0.3)"
              tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
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
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
