"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle } from "react-leaflet";
import L from "leaflet";

// Fix Leaflet default icon issues in Next.js
const iconUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png';
const iconRetinaUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png';
const shadowUrl = 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png';

const DefaultIcon = L.icon({
  iconUrl,
  iconRetinaUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
  shadowSize: [41, 41]
});

// Custom icons
const TigerIcon = L.icon({
  iconUrl,
  iconRetinaUrl,
  shadowUrl,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
  className: 'hue-rotate-[180deg] filter' // Makes it orange-ish
});

// Mock camera locations (Nagpur region approximate)
const cameras = [
  { id: "CAM001", name: "North Gate", lat: 21.1458, lon: 79.0882, status: "offline", recent: "None" },
  { id: "CAM007", name: "Waterhole Alpha", lat: 21.1500, lon: 79.0950, status: "active", recent: "Human + Vehicle" },
  { id: "CAM012", name: "Tiger Trail East", lat: 21.1350, lon: 79.1000, status: "active", recent: "Bengal Tiger", isTiger: true },
  { id: "CAM022", name: "Village Border South", lat: 21.1200, lon: 79.0800, status: "active", recent: "Wild Boar" },
];

export default function LiveMap() {
  // Center near Nagpur
  const center = [21.1400, 79.0900];

  return (
    <MapContainer 
      center={center as [number, number]} 
      zoom={13} 
      style={{ height: "100%", width: "100%", background: "#1c2720" }}
      zoomControl={false}
    >
      {/* Dark theme styled map tiles */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      {/* Core Zone overlay */}
      <Circle 
        center={[21.1420, 79.0950]} 
        radius={1500} 
        pathOptions={{ color: '#338a3d', fillColor: '#338a3d', fillOpacity: 0.1, weight: 2, dashArray: '5, 5' }} 
      />
      
      {/* Restricted Zone overlay */}
      <Circle 
        center={[21.1500, 79.0950]} 
        radius={800} 
        pathOptions={{ color: '#d32f2f', fillColor: '#d32f2f', fillOpacity: 0.1, weight: 1 }} 
      />

      {/* Cameras */}
      {cameras.map((cam) => (
        <Marker 
          key={cam.id} 
          position={[cam.lat, cam.lon]}
          icon={cam.isTiger ? TigerIcon : DefaultIcon}
        >
          <Popup>
            <div className="p-1 min-w-40">
              <h3 className="font-bold text-sm mb-1">{cam.name} ({cam.id})</h3>
              <div className="flex items-center gap-2 mb-2 text-xs">
                <span className={`w-2 h-2 rounded-full ${cam.status === 'active' ? 'bg-success' : 'bg-warning'}`}></span>
                <span className="capitalize">{cam.status}</span>
              </div>
              <div className="text-xs text-foreground/70">
                Recent: <span className={cam.isTiger ? "text-accent-500 font-bold" : "text-white"}>{cam.recent}</span>
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
