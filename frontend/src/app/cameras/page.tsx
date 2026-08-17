"use client";

import { Camera, Settings, Activity } from "lucide-react";

const cameras = [
  { id: "CAM001", name: "North Gate", status: "offline", battery: "0%", images: 1240, lastActive: "1 day ago" },
  { id: "CAM007", name: "Waterhole Alpha", status: "active", battery: "84%", images: 4532, lastActive: "5 min ago" },
  { id: "CAM012", name: "Tiger Trail East", status: "active", battery: "62%", images: 2108, lastActive: "2 hrs ago" },
  { id: "CAM022", name: "Village Border South", status: "active", battery: "91%", images: 512, lastActive: "10 min ago" },
];

export default function Cameras() {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Cameras</h1>
        <p className="text-foreground/70">Manage your camera trap network.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {cameras.map(cam => (
          <div key={cam.id} className="glass-card rounded-xl p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="font-bold text-lg text-white">{cam.name}</h3>
                <p className="text-xs font-mono text-foreground/50">{cam.id}</p>
              </div>
              <div className={`px-2 py-1 rounded text-xs font-medium ${cam.status === 'active' ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'}`}>
                {cam.status.toUpperCase()}
              </div>
            </div>
            
            <div className="space-y-2 text-sm text-foreground/70">
              <div className="flex justify-between">
                <span>Battery:</span>
                <span className={cam.battery === "0%" ? "text-danger" : "text-white"}>{cam.battery}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Images:</span>
                <span className="text-white">{cam.images.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Last Active:</span>
                <span className="text-white">{cam.lastActive}</span>
              </div>
            </div>
            
            <div className="mt-6 flex gap-2">
              <button className="flex-1 bg-surface-200 hover:bg-surface-100 text-white py-2 rounded-lg text-xs font-medium transition-colors flex justify-center items-center gap-2">
                <Settings size={14} /> Configure
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
