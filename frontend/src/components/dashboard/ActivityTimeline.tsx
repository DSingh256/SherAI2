"use client";

import React from "react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

interface ActivityTimelineProps {
  timeline?: Array<{ date: string; count: number }>;
}

const defaultTimeline = [
  { date: "Aug 11", count: 18 },
  { date: "Aug 12", count: 24 },
  { date: "Aug 13", count: 19 },
  { date: "Aug 14", count: 32 },
  { date: "Aug 15", count: 41 },
  { date: "Aug 16", count: 35 },
  { date: "Aug 17", count: 28 },
];

export const ActivityTimeline: React.FC<ActivityTimelineProps> = ({ timeline = defaultTimeline }) => {
  const chartData = timeline.length > 0 ? timeline : defaultTimeline;

  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-5 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="font-semibold text-white text-base">Wildlife Activity Timeline</h3>
          <p className="text-xs text-foreground/60">Capture frequency over the past 7 days</p>
        </div>
      </div>

      <div className="flex-1 min-h-[240px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 15, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="date"
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
            />
            <Area
              type="monotone"
              dataKey="count"
              stroke="#10b981"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#colorCount)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
