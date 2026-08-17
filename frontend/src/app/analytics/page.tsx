"use client";

import { BarChart3, TrendingUp, AlertTriangle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const threatData = [
  { zone: "Core Zone", threats: 12 },
  { zone: "Buffer North", threats: 18 },
  { zone: "Tiger Trail", threats: 5 },
  { zone: "Village Border", threats: 24 },
];

export default function Analytics() {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Analytics</h1>
        <p className="text-foreground/70">Deep insights into wildlife populations and threat patterns.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-medium text-white mb-6 flex items-center gap-2">
            <AlertTriangle className="text-warning" size={18} />
            Threat Heatmap (by Zone)
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={threatData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="zone" stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} tickLine={false} axisLine={false} />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--color-surface-100)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                />
                <Bar dataKey="threats" fill="var(--color-danger)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="glass-card rounded-xl p-6 flex items-center justify-center">
          <div className="text-center text-foreground/50">
            <TrendingUp size={48} className="mx-auto mb-4 opacity-50" />
            <p>Population trend models generating...</p>
            <p className="text-sm mt-2">Requires 30 days of data for baseline.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
