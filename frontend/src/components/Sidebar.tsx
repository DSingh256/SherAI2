"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  Map as MapIcon, 
  CheckSquare, 
  BarChart3, 
  Camera, 
  Bell, 
  History,
  Fingerprint
} from "lucide-react";

const NAV_ITEMS = [
  { name: "Dashboard", path: "/", icon: LayoutDashboard },
  { name: "Review Queue", path: "/review", icon: CheckSquare, badge: "12" },
  { name: "Live Map", path: "/map", icon: MapIcon },
  { name: "Tiger Re-ID", path: "/re-id", icon: Fingerprint, badge: "New" },
  { name: "Analytics", path: "/analytics", icon: BarChart3 },
  { name: "Cameras", path: "/cameras", icon: Camera },
  { name: "Alerts", path: "/alerts", icon: Bell, alert: true },
  { name: "Audit Trail", path: "/audit", icon: History },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 h-screen fixed left-0 top-0 glass-panel border-r border-r-white/5 flex flex-col z-40">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 rounded bg-primary-500 flex items-center justify-center font-bold text-white shadow-[0_0_15px_rgba(51,138,61,0.5)]">
            VR
          </div>
          <div>
            <h1 className="font-bold text-xl tracking-tight text-white">VanRakshak AI</h1>
            <p className="text-xs text-primary-300">Wildlife Intelligence</p>
          </div>
        </div>

        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = pathname === item.path;
            const Icon = item.icon;
            
            return (
              <Link 
                key={item.path} 
                href={item.path}
                className={`flex items-center justify-between px-3 py-2.5 rounded-lg transition-all duration-200 group ${
                  isActive 
                    ? "bg-primary-900/50 text-white shadow-inner border border-primary-500/20" 
                    : "text-foreground/70 hover:bg-surface-200 hover:text-white"
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon size={18} className={`${isActive ? "text-primary-400" : "text-foreground/50 group-hover:text-primary-400"} transition-colors`} />
                  <span className="font-medium text-sm">{item.name}</span>
                </div>
                
                {item.badge && (
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                    isActive ? "bg-primary-500 text-white" : "bg-surface-100 text-foreground/70 group-hover:bg-primary-500/30"
                  }`}>
                    {item.badge}
                  </span>
                )}
                
                {item.alert && (
                  <span className="w-2 h-2 rounded-full bg-accent-500 shadow-[0_0_8px_rgba(245,124,0,0.8)] animate-pulse"></span>
                )}
              </Link>
            );
          })}
        </nav>
      </div>
      
      <div className="mt-auto p-6">
        <div className="bg-surface-100 rounded-lg p-4 border border-white/5">
          <div className="text-xs text-foreground/50 mb-2">System Status</div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-success"></div>
            <span className="text-sm font-medium text-success">All Systems Operational</span>
          </div>
        </div>
      </div>
    </div>
  );
}
