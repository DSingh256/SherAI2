"use client";

import React from "react";
import { Camera, MapPin, Flame } from "lucide-react";

interface HotspotItem {
  camera_id: string;
  total_captures: number;
  tiger_sightings: number;
  top_species: string;
  status: string;
}

interface CameraHotspotsProps {
  hotspots?: HotspotItem[];
}

const defaultHotspots: HotspotItem[] = [
  { camera_id: "CAM007_Waterhole_Alpha", total_captures: 64, tiger_sightings: 6, top_species: "Bengal Tiger", status: "active" },
  { camera_id: "CAM012_Tiger_Corridor_East", total_captures: 48, tiger_sightings: 4, top_species: "Bengal Tiger", status: "active" },
  { camera_id: "CAM003_Grassland_North", total_captures: 39, tiger_sightings: 1, top_species: "Spotted Deer", status: "active" },
  { camera_id: "CAM019_Dense_Canopy_South", total_captures: 28, tiger_sightings: 0, top_species: "Indian Leopard", status: "active" },
  { camera_id: "CAM022_Buffer_Zone_Village", total_captures: 22, tiger_sightings: 1, top_species: "Wild Boar", status: "active" },
];

export const CameraHotspots: React.FC<CameraHotspotsProps> = ({ hotspots = defaultHotspots }) => {
  const items = hotspots.length > 0 ? hotspots : defaultHotspots;

  return (
    <div className="rounded-xl border border-white/10 bg-surface-100/70 p-5 space-y-4">
      <div>
        <h3 className="font-semibold text-white text-base flex items-center gap-2">
          <Flame size={16} className="text-amber-400" />
          Camera Trap Activity Hotspots
        </h3>
        <p className="text-xs text-foreground/60">Ranked by wildlife traffic & corridor density</p>
      </div>

      <div className="space-y-2">
        {items.map((cam, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-white/5 hover:bg-surface-200/80 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-surface-300 flex items-center justify-center text-primary-400 font-mono text-xs font-bold">
                #{idx + 1}
              </div>
              <div>
                <h4 className="text-xs font-semibold text-white font-mono">{cam.camera_id}</h4>
                <p className="text-[11px] text-foreground/60">Primary: {cam.top_species}</p>
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono text-right">
              {cam.tiger_sightings > 0 && (
                <span className="px-2 py-0.5 rounded text-[11px] bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  🐅 {cam.tiger_sightings} sightings
                </span>
              )}
              <div>
                <span className="font-bold text-white">{cam.total_captures}</span>
                <span className="text-foreground/50 text-[10px] ml-1">captures</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
