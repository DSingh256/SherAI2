"use client";

import React from "react";

interface ConfidenceBadgeProps {
  confidence: number;
  level?: string;
  size?: "sm" | "md" | "lg";
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  confidence,
  level,
  size = "md",
}) => {
  const percentage = (confidence * 100).toFixed(1);
  const derivedLevel =
    level || (confidence >= 0.9 ? "high" : confidence >= 0.6 ? "medium" : "low");

  const colorStyles = {
    high: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    medium: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    low: "bg-rose-500/20 text-rose-400 border-rose-500/30",
  }[derivedLevel];

  const sizeStyles = {
    sm: "px-2 py-0.5 text-xs font-mono",
    md: "px-2.5 py-1 text-sm font-mono font-medium",
    lg: "px-3.5 py-1.5 text-base font-mono font-semibold",
  }[size];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border ${colorStyles} ${sizeStyles}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          derivedLevel === "high"
            ? "bg-emerald-400 animate-pulse"
            : derivedLevel === "medium"
            ? "bg-amber-400"
            : "bg-rose-400"
        }`}
      />
      <span>{percentage}%</span>
      <span className="text-[10px] uppercase tracking-wider opacity-75">
        ({derivedLevel})
      </span>
    </span>
  );
};
