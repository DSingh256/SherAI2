"use client";

import React from "react";
import { Camera, CheckCircle2, AlertTriangle, ShieldCheck, Activity, Eye } from "lucide-react";
import Link from "next/link";

interface StatsOverviewProps {
  stats: {
    total_images?: number;
    animals_detected?: number;
    species_detected?: number;
    tigers_detected?: number;
    auto_accepted?: number;
    pending_review?: number;
    human_reviewed?: number;
    active_cameras?: number;
    automation_rate?: number;
  };
  isLoading?: boolean;
}

export const StatsOverview: React.FC<StatsOverviewProps> = ({ stats = {}, isLoading = false }) => {
  const cards = [
    {
      title: "Total Captures",
      value: stats.total_images ?? 184,
      subtext: `${stats.animals_detected ?? 142} animals detected`,
      icon: Camera,
      color: "text-primary-400",
      bgGlow: "bg-primary-500/10 group-hover:bg-primary-500/20",
    },
    {
      title: "Auto-Accepted",
      value: stats.auto_accepted ?? 128,
      subtext: `${stats.automation_rate ?? 69.5}% automation rate`,
      icon: CheckCircle2,
      color: "text-emerald-400",
      bgGlow: "bg-emerald-500/10 group-hover:bg-emerald-500/20",
    },
    {
      title: "Pending Review",
      value: stats.pending_review ?? 14,
      subtext: "Needs human verification",
      icon: AlertTriangle,
      color: "text-amber-400",
      bgGlow: "bg-amber-500/10 group-hover:bg-amber-500/20",
      link: "/review",
      linkText: "Go to review queue →",
    },
    {
      title: "Tigers Sighted",
      value: stats.tigers_detected ?? 12,
      subtext: "Priority tracking active",
      icon: Activity,
      color: "text-amber-500",
      bgGlow: "bg-amber-500/20 group-hover:bg-amber-500/30",
      link: "/reid",
      linkText: "View Tiger Re-ID →",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className="rounded-xl border border-white/10 bg-surface-100/70 p-5 relative overflow-hidden group transition-all hover:border-white/20"
          >
            <div
              className={`absolute -right-6 -top-6 w-24 h-24 rounded-full blur-xl transition-all ${card.bgGlow}`}
            />
            <div className="flex items-start justify-between mb-3 relative z-10">
              <div>
                <p className="text-xs font-medium text-foreground/60 uppercase tracking-wider mb-1">
                  {card.title}
                </p>
                <h3 className="text-2xl font-bold text-white font-mono">
                  {isLoading ? "..." : card.value.toLocaleString()}
                </h3>
              </div>
              <div className={`p-2.5 rounded-lg bg-surface-200/80 ${card.color} border border-white/5`}>
                <Icon size={18} />
              </div>
            </div>

            <div className="flex items-center justify-between text-xs pt-1 border-t border-white/5 relative z-10">
              <span className="text-foreground/50">{card.subtext}</span>
              {card.link && (
                <Link href={card.link} className={`${card.color} hover:underline font-medium text-[11px]`}>
                  {card.linkText}
                </Link>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
