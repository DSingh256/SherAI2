"use client";

import { useState } from "react";
import { Check, X, AlertTriangle, Search, Filter } from "lucide-react";

// Mock data for review queue
const mockQueue = [
  {
    id: "img_001",
    camera: "CAM007 (Waterhole Alpha)",
    timestamp: "2024-03-10 18:45:22",
    prediction: "Indian Leopard",
    confidence: 0.58,
    reasoning: [
      "✓ Animal detected with 92% confidence",
      "⚠ SpeciesNet predicts Indian Leopard at 58%",
      "⚠ OpenCLIP disagrees — predicts Jungle Cat",
      "✓ Models show low agreement — verification recommended"
    ],
    image: "https://images.unsplash.com/photo-1544641957-3f360c497424?w=800&q=80"
  },
  {
    id: "img_002",
    camera: "CAM012 (Tiger Trail East)",
    timestamp: "2024-03-10 03:15:00",
    prediction: "Bengal Tiger",
    confidence: 0.72,
    reasoning: [
      "✓ Animal detected with 98% confidence",
      "✓ SpeciesNet predicts Bengal Tiger at 72%",
      "⚠ Night vision image quality is acceptable",
      "⚠ Unusual activity time (03:00) for Tiger"
    ],
    image: "https://images.unsplash.com/photo-1579753767215-6217435f3032?w=800&q=80"
  },
  {
    id: "img_003",
    camera: "CAM022 (Village Border)",
    timestamp: "2024-03-09 22:10:05",
    prediction: "Human",
    confidence: 0.65,
    reasoning: [
      "✓ Human detected with 85% confidence",
      "⚠ Activity detected at unusual hour (22:00)",
      "⚠ Human activity in restricted zone"
    ],
    image: "https://images.unsplash.com/photo-1534067783941-51c9c23ecefd?w=800&q=80"
  }
];

export default function ReviewQueue() {
  const [queue, setQueue] = useState(mockQueue);
  const [selectedImage, setSelectedImage] = useState(mockQueue[0]);

  const handleDecision = (id: string, decision: string) => {
    // In a real app, this would POST to /api/review/submit
    const newQueue = queue.filter(item => item.id !== id);
    setQueue(newQueue);
    if (newQueue.length > 0) {
      setSelectedImage(newQueue[0]);
    } else {
      // @ts-expect-error - Next.js state update for null vs mockQueue type
      setSelectedImage(null);
    }
  };

  return (
    <div className="space-y-6 h-[calc(100vh-8rem)] flex flex-col">
      <header className="flex justify-between items-end mb-2">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-2">Review Queue</h1>
          <p className="text-foreground/70">Human verification for uncertain AI classifications.</p>
        </div>
        <div className="flex gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-foreground/50" size={16} />
            <input 
              type="text" 
              placeholder="Search ID or camera..." 
              className="pl-10 pr-4 py-2 rounded-lg text-sm w-64"
            />
          </div>
          <button className="glass-card px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-surface-200 transition-colors">
            <Filter size={16} />
            <span className="text-sm font-medium">Filter</span>
          </button>
        </div>
      </header>

      {queue.length === 0 ? (
        <div className="flex-1 glass-card rounded-xl flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 rounded-full bg-success/20 flex items-center justify-center text-success mb-4">
            <Check size={32} />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">All Caught Up!</h2>
          <p className="text-foreground/60 max-w-md">
            The review queue is empty. The AI is confident in all recent classifications.
          </p>
        </div>
      ) : (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-0">
          {/* Queue List */}
          <div className="glass-card rounded-xl overflow-hidden flex flex-col border border-white/5">
            <div className="p-4 border-b border-white/5 bg-surface-100/50">
              <h3 className="font-medium text-white flex justify-between">
                <span>Pending Verification</span>
                <span className="bg-warning/20 text-warning px-2 py-0.5 rounded text-xs">{queue.length}</span>
              </h3>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {queue.map(item => (
                <div 
                  key={item.id}
                  onClick={() => setSelectedImage(item)}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    selectedImage?.id === item.id 
                      ? "bg-primary-900/40 border border-primary-500/30" 
                      : "hover:bg-surface-200 border border-transparent"
                  }`}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="text-xs font-mono text-foreground/50">{item.id}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${
                      item.prediction === 'Bengal Tiger' ? 'bg-accent-500/20 text-accent-400' :
                      item.prediction === 'Human' ? 'bg-danger/20 text-danger' :
                      'bg-surface-200 text-foreground/70'
                    }`}>
                      {item.prediction} ({(item.confidence * 100).toFixed(0)}%)
                    </span>
                  </div>
                  <div className="font-medium text-sm text-white truncate">{item.camera}</div>
                  <div className="text-xs text-foreground/60 mt-1">{item.timestamp}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Review Panel */}
          <div className="lg:col-span-2 glass-card rounded-xl overflow-hidden flex flex-col border border-white/5 relative">
            {selectedImage && (
              <>
                <div className="h-2/5 bg-surface-200 relative group">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={selectedImage.image} 
                    alt="Camera trap capture" 
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
                    <div className="text-white text-xs">
                      <span className="opacity-70">Raw Path:</span> /storage/raw/img_{selectedImage.id}.jpg
                    </div>
                  </div>
                </div>
                
                <div className="flex-1 p-6 flex flex-col overflow-y-auto">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-2xl font-bold text-white mb-1">AI Prediction: {selectedImage.prediction}</h2>
                      <div className="flex gap-4 text-sm text-foreground/60">
                        <span>{selectedImage.camera}</span>
                        <span>•</span>
                        <span>{selectedImage.timestamp}</span>
                      </div>
                    </div>
                    <div className="flex flex-col items-end">
                      <div className="text-3xl font-light text-primary-400">
                        {(selectedImage.confidence * 100).toFixed(1)}<span className="text-lg text-primary-400/50">%</span>
                      </div>
                      <span className="text-xs text-foreground/50 uppercase tracking-wider">Confidence</span>
                    </div>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-foreground/70 uppercase tracking-wider mb-3">AI Reasoning</h3>
                    <div className="space-y-2 bg-surface-100 rounded-lg p-4 border border-white/5">
                      {selectedImage.reasoning.map((reason, idx) => (
                        <div key={idx} className="flex gap-2 text-sm">
                          <span className={
                            reason.startsWith('✓') ? 'text-success' : 
                            reason.startsWith('⚠') ? 'text-warning' : 'text-danger'
                          }>
                            {reason.charAt(0)}
                          </span>
                          <span className="text-foreground/80">{reason.substring(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mt-auto border-t border-white/10 pt-6">
                    <h3 className="text-sm font-medium text-foreground/70 uppercase tracking-wider mb-4">Human Verification</h3>
                    
                    <div className="flex gap-3 mb-4">
                      <select className="flex-1 p-3 rounded-lg text-sm bg-surface-100 border border-white/10 text-white">
                        <option value={selectedImage.prediction}>Confirm: {selectedImage.prediction}</option>
                        <option value="Bengal Tiger">Correct to: Bengal Tiger</option>
                        <option value="Indian Leopard">Correct to: Indian Leopard</option>
                        <option value="Wild Boar">Correct to: Wild Boar</option>
                        <option value="Human">Correct to: Human (Poacher/Patrol)</option>
                        <option value="Vehicle">Correct to: Vehicle</option>
                        <option value="Empty">Correct to: Empty (False Alarm)</option>
                      </select>
                      
                      <input 
                        type="text" 
                        placeholder="Add review notes (optional)..." 
                        className="flex-1 p-3 rounded-lg text-sm"
                      />
                    </div>
                    
                    <div className="flex gap-3">
                      <button 
                        onClick={() => handleDecision(selectedImage.id, "confirm")}
                        className="flex-1 bg-primary-600 hover:bg-primary-500 text-white py-3 rounded-lg font-medium transition-colors flex justify-center items-center gap-2"
                      >
                        <Check size={18} /> Confirm Classification
                      </button>
                      <button 
                        onClick={() => handleDecision(selectedImage.id, "reject")}
                        className="px-6 bg-surface-200 hover:bg-danger/20 hover:text-danger text-foreground py-3 rounded-lg font-medium transition-colors border border-transparent hover:border-danger/30"
                      >
                        <AlertTriangle size={18} />
                      </button>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
