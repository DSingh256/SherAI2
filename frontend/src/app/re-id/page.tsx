"use client";

import { Fingerprint, CheckCircle2, XCircle } from "lucide-react";
import Image from "next/image";

export default function ReIdentification() {
  return (
    <div className="space-y-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Tiger Re-identification</h1>
        <p className="text-foreground/70">Potential tiger matches based on stripe pattern embeddings.</p>
      </header>

      <div className="grid grid-cols-1 gap-6 max-w-5xl">
        <div className="glass-card rounded-xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-lg text-white">Match Confidence: 94.2%</h3>
            <span className="bg-warning/20 text-warning px-3 py-1 rounded-lg text-sm font-medium">Pending Verification</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div className="space-y-3">
              <div className="text-sm font-medium text-foreground/70 text-center uppercase tracking-wider">Current Detection</div>
              <div className="aspect-video bg-surface-200 rounded-lg relative overflow-hidden group">
                <img src="https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=800&q=80" alt="Current tiger" className="w-full h-full object-cover" />
                <div className="absolute inset-0 border-2 border-primary-500 rounded-lg pointer-events-none"></div>
              </div>
              <div className="flex justify-between text-sm text-foreground/60">
                <span>CAM012 (Tiger Trail East)</span>
                <span>Just now</span>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="text-sm font-medium text-foreground/70 text-center uppercase tracking-wider">Historical Match</div>
              <div className="aspect-video bg-surface-200 rounded-lg relative overflow-hidden group">
                <img src="https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=800&q=80" alt="Matched tiger" className="w-full h-full object-cover filter brightness-90 grayscale-[0.2]" />
              </div>
              <div className="flex justify-between text-sm text-foreground/60">
                <span>CAM007 (Waterhole Alpha)</span>
                <span>12 days ago</span>
              </div>
            </div>
          </div>
          
          <div className="flex justify-center items-center gap-4 border-t border-white/5 pt-6">
            <button className="flex items-center gap-2 bg-primary-600 hover:bg-primary-500 text-white px-6 py-2.5 rounded-lg font-medium transition-colors">
              <CheckCircle2 size={18} /> Confirm Match (T-42)
            </button>
            <button className="flex items-center gap-2 bg-surface-200 hover:bg-surface-100 text-white px-6 py-2.5 rounded-lg font-medium transition-colors">
              <XCircle size={18} /> Not a Match
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
