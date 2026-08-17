"use client";

import { Bell, AlertTriangle } from "lucide-react";

export default function Alerts() {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">System Alerts</h1>
        <p className="text-foreground/70">Real-time threat and conservation alerts.</p>
      </header>

      <div className="glass-card rounded-xl p-6 max-w-4xl">
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-danger/10 border border-danger/30 flex gap-4">
            <AlertTriangle className="text-danger flex-shrink-0" />
            <div>
              <h3 className="font-bold text-white mb-1">Human + Vehicle Activity at 3AM</h3>
              <p className="text-sm text-foreground/70 mb-2">Multiple humans and a vehicle detected in Core Zone during restricted hours. Ranger verification required immediately.</p>
              <div className="text-xs text-foreground/50">CAM007 • 2 hours ago</div>
            </div>
            <button className="ml-auto self-start bg-danger/20 text-danger px-3 py-1.5 rounded text-sm hover:bg-danger hover:text-white transition-colors">
              Dispatch Patrol
            </button>
          </div>
          
          <div className="p-4 rounded-lg bg-accent-500/10 border border-accent-500/30 flex gap-4">
            <Bell className="text-accent-500 flex-shrink-0" />
            <div>
              <h3 className="font-bold text-white mb-1">Tiger Detected</h3>
              <p className="text-sm text-foreground/70 mb-2">Bengal Tiger detected on Tiger Trail East.</p>
              <div className="text-xs text-foreground/50">CAM012 • 5 hours ago</div>
            </div>
          </div>
          
          <div className="p-4 rounded-lg bg-warning/10 border border-warning/30 flex gap-4">
            <AlertTriangle className="text-warning flex-shrink-0" />
            <div>
              <h3 className="font-bold text-white mb-1">Camera Offline</h3>
              <p className="text-sm text-foreground/70 mb-2">No heartbeat received from North Gate camera for 24 hours.</p>
              <div className="text-xs text-foreground/50">CAM001 • 1 day ago</div>
            </div>
            <button className="ml-auto self-start bg-surface-200 text-white px-3 py-1.5 rounded text-sm hover:bg-surface-100 transition-colors">
              Acknowledge
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
