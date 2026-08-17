"use client";

import { useEffect, useState } from "react";
import { 
  Camera, 
  CheckCircle2, 
  AlertTriangle, 
  HelpCircle,
  Activity,
  ArrowRight
} from "lucide-react";
import Link from "next/link";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line
} from "recharts";

// Mock data as fallback
const mockTimelineData = [
  { date: "Mon", count: 12 },
  { date: "Tue", count: 19 },
  { date: "Wed", count: 15 },
  { date: "Thu", count: 22 },
  { date: "Fri", count: 30 },
  { date: "Sat", count: 28 },
  { date: "Sun", count: 24 },
];

const mockSpeciesData = [
  { name: "Spotted Deer", count: 45 },
  { name: "Sambar", count: 32 },
  { name: "Wild Boar", count: 28 },
  { name: "Langur", count: 15 },
  { name: "Bengal Tiger", count: 8 },
  { name: "Leopard", count: 5 },
];

export default function Dashboard() {
  const [stats, setStats] = useState({
    total_images: 150,
    auto_accepted: 102,
    pending_review: 35,
    human_reviewed: 13,
    tigers_detected: 8
  });
  
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // In a real scenario, this would fetch from the backend
    // fetch('http://127.0.0.1:8000/api/analytics/overview')
    //   .then(res => res.json())
    //   .then(data => setStats(data.data))
    //   .catch(console.error);
      
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="space-y-6">
      <header className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Overview</h1>
          <p className="text-foreground/70">Central intelligence for all camera trap networks.</p>
        </div>
        <div className="flex gap-4">
          <div className="glass-card px-4 py-2 rounded-lg flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
            <span className="text-sm font-medium">Pipeline Active</span>
          </div>
        </div>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Stat Card 1 */}
        <div className="glass-card rounded-xl p-6 relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-primary-500/10 rounded-full blur-xl group-hover:bg-primary-500/20 transition-all"></div>
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-foreground/60 mb-1">Total Processed</p>
              <h3 className="text-3xl font-bold text-white">
                {isLoading ? "..." : stats.total_images}
              </h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-surface-200 flex items-center justify-center text-primary-400">
              <Camera size={20} />
            </div>
          </div>
          <div className="flex items-center text-xs">
            <span className="text-success font-medium">+12%</span>
            <span className="text-foreground/50 ml-2">from last week</span>
          </div>
        </div>

        {/* Stat Card 2 */}
        <div className="glass-card rounded-xl p-6 relative overflow-hidden group">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-success/10 rounded-full blur-xl group-hover:bg-success/20 transition-all"></div>
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-foreground/60 mb-1">Auto-Accepted</p>
              <h3 className="text-3xl font-bold text-white">
                {isLoading ? "..." : stats.auto_accepted}
              </h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-surface-200 flex items-center justify-center text-success">
              <CheckCircle2 size={20} />
            </div>
          </div>
          <div className="flex items-center text-xs">
            <span className="text-foreground/70 font-medium">
              {Math.round((stats.auto_accepted / stats.total_images) * 100)}% automation rate
            </span>
          </div>
        </div>

        {/* Stat Card 3 */}
        <div className="glass-card rounded-xl p-6 relative overflow-hidden group border-warning/30 shadow-[0_0_15px_rgba(252,149,56,0.05)]">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-warning/10 rounded-full blur-xl group-hover:bg-warning/20 transition-all"></div>
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-foreground/60 mb-1">Pending Review</p>
              <h3 className="text-3xl font-bold text-white">
                {isLoading ? "..." : stats.pending_review}
              </h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-warning/20 flex items-center justify-center text-warning">
              <HelpCircle size={20} />
            </div>
          </div>
          <Link href="/review" className="flex items-center text-xs text-warning hover:text-warning/80 font-medium group/link">
            Go to review queue
            <ArrowRight size={12} className="ml-1 transition-transform group-hover/link:translate-x-1" />
          </Link>
        </div>

        {/* Stat Card 4 */}
        <div className="glass-card rounded-xl p-6 relative overflow-hidden group border-accent-500/30 bg-accent-500/5">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-accent-500/10 rounded-full blur-xl group-hover:bg-accent-500/20 transition-all"></div>
          <div className="flex justify-between items-start mb-4">
            <div>
              <p className="text-sm font-medium text-accent-500 mb-1">Tigers Detected</p>
              <h3 className="text-3xl font-bold text-white">
                {isLoading ? "..." : stats.tigers_detected}
              </h3>
            </div>
            <div className="w-10 h-10 rounded-lg bg-accent-500/20 flex items-center justify-center text-accent-500">
              <Activity size={20} />
            </div>
          </div>
          <Link href="/map" className="flex items-center text-xs text-accent-400 hover:text-accent-300 font-medium group/link">
            View on map
            <ArrowRight size={12} className="ml-1 transition-transform group-hover/link:translate-x-1" />
          </Link>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Activity Timeline */}
        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-medium text-white mb-6">Activity Timeline (7 days)</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockTimelineData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <Line type="monotone" dataKey="count" stroke="var(--color-primary-400)" strokeWidth={3} dot={{ fill: 'var(--color-background)', strokeWidth: 2 }} activeDot={{ r: 6, fill: 'var(--color-primary-400)' }} />
                <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="date" stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} tickLine={false} axisLine={false} />
                <YAxis stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--color-surface-100)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                  itemStyle={{ color: 'var(--color-primary-400)' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Species Distribution */}
        <div className="glass-card rounded-xl p-6">
          <h3 className="text-lg font-medium text-white mb-6">Species Distribution</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockSpeciesData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
                <XAxis type="number" stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }} tickLine={false} axisLine={false} />
                <YAxis dataKey="name" type="category" stroke="rgba(255,255,255,0.3)" tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 12 }} tickLine={false} axisLine={false} width={100} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--color-surface-100)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                />
                <Bar dataKey="count" fill="var(--color-primary-500)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="glass-card rounded-xl p-6 mt-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-medium text-white">Recent Alerts</h3>
          <Link href="/alerts" className="text-sm text-primary-400 hover:text-primary-300 font-medium">View all</Link>
        </div>
        
        <div className="space-y-3">
          {[
            { id: 1, title: "Human + Vehicle Activity at 3AM", zone: "Core Zone", time: "2 hours ago", type: "threat" },
            { id: 2, title: "Tiger Detected", zone: "Tiger Trail East", time: "5 hours ago", type: "tiger" },
            { id: 3, title: "Camera Offline", zone: "North Gate", time: "1 day ago", type: "system" }
          ].map((alert) => (
            <div key={alert.id} className="flex items-center justify-between p-4 rounded-lg bg-surface-100 border border-white/5 hover:bg-surface-200 transition-colors">
              <div className="flex items-center gap-4">
                <div className={`w-2 h-10 rounded-full ${
                  alert.type === 'threat' ? 'bg-danger' : 
                  alert.type === 'tiger' ? 'bg-accent-500' : 'bg-warning'
                }`}></div>
                <div>
                  <h4 className="font-medium text-white">{alert.title}</h4>
                  <p className="text-sm text-foreground/60">{alert.zone}</p>
                </div>
              </div>
              <div className="text-sm text-foreground/50">{alert.time}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
