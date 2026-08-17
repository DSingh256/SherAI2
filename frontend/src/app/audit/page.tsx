"use client";

import { History, ShieldCheck, User } from "lucide-react";

export default function AuditTrail() {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Audit Trail</h1>
        <p className="text-foreground/70">Immutable record of all AI decisions and human interventions.</p>
      </header>

      <div className="glass-card rounded-xl overflow-hidden max-w-5xl">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-foreground/50 uppercase bg-surface-100/50">
              <tr>
                <th className="px-6 py-4">Timestamp</th>
                <th className="px-6 py-4">Image ID</th>
                <th className="px-6 py-4">Event Type</th>
                <th className="px-6 py-4">Action/Details</th>
                <th className="px-6 py-4">User</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {[
                { time: "2024-03-10 18:45:23", id: "img_001", type: "human_review", action: "Corrected classification: Leopard → Jungle Cat", user: "Ranger Sharma" },
                { time: "2024-03-10 18:45:22", id: "img_001", type: "decision_engine", action: "Routed to HUMAN_REVIEW (Confidence 58%)", user: "System" },
                { time: "2024-03-10 18:45:21", id: "img_001", type: "privacy_protection", action: "Applied face blur (1 face detected)", user: "System" },
                { time: "2024-03-10 18:45:20", id: "img_001", type: "quality_gate", action: "Passed quality check (Score 0.85)", user: "System" },
                { time: "2024-03-10 03:15:02", id: "img_002", type: "decision_engine", action: "AUTO_ACCEPT classification: Bengal Tiger", user: "System" },
              ].map((log, i) => (
                <tr key={i} className="hover:bg-surface-100 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-foreground/70 font-mono text-xs">{log.time}</td>
                  <td className="px-6 py-4 whitespace-nowrap font-medium text-white">{log.id}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded text-xs ${
                      log.type === 'human_review' ? 'bg-primary-500/20 text-primary-400' :
                      log.type === 'decision_engine' ? 'bg-surface-200 text-white' :
                      'bg-surface-200/50 text-foreground/70'
                    }`}>
                      {log.type}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-foreground/80">{log.action}</td>
                  <td className="px-6 py-4 whitespace-nowrap flex items-center gap-2 text-foreground/60">
                    {log.user === 'System' ? <ShieldCheck size={14} className="text-primary-500" /> : <User size={14} className="text-accent-400" />}
                    {log.user}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
