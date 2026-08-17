"use client";

import React, { useState, useCallback, useRef } from "react";
import {
  Upload,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Eye,
  Zap,
  Shield,
  Brain,
  Camera,
  MapPin,
  ChevronDown,
  ChevronUp,
  Sparkles,
  ScanEye,
  Layers,
  FileCheck,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

// Pipeline stage labels
const PIPELINE_STAGES = [
  { key: "upload", label: "Image Upload", icon: Upload },
  { key: "quality", label: "Quality Gate", icon: Shield },
  { key: "megadetector", label: "MegaDetector V6", icon: ScanEye },
  { key: "speciesnet", label: "SpeciesNet", icon: Brain },
  { key: "openclip", label: "OpenCLIP Verify", icon: Eye },
  { key: "segmentation", label: "SAM2 Segment", icon: Layers },
  { key: "decision", label: "Decision Engine", icon: Zap },
];

type AnalysisResult = {
  image_id: string;
  pipeline_success: boolean;
  pipeline_time_ms: number;
  image: {
    camera_id: string;
    timestamp: string;
    location: string | null;
    width: number | null;
    height: number | null;
    file_size: number;
    image_path: string | null;
  };
  quality: {
    status: string;
    score: number | null;
    blur_score: number | null;
    brightness: number | null;
    contrast: number | null;
    passed: boolean;
  };
  detections: Array<{
    id: string;
    object_type: string;
    confidence: number;
    bbox: { x_min: number; y_min: number; x_max: number; y_max: number };
    crop_path: string | null;
    classifications: Array<{
      species: string;
      confidence: number;
      alternatives: Array<{ species: string; confidence: number }>;
      model_name: string;
    }>;
  }>;
  verification: {
    primary_prediction: string;
    confidence: number;
    semantic_scores: Record<string, number>;
    model_name: string;
  } | null;
  decision: {
    species: string | null;
    confidence: number;
    decision: string;
    confidence_level: string;
    reasoning: string[];
    signals: Record<string, unknown>;
    is_tiger: boolean;
  } | null;
};

function formatBytes(bytes: number) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function ConfidenceBar({
  value,
  color = "primary",
}: {
  value: number;
  color?: string;
}) {
  const pct = Math.round(value * 100);
  const clr =
    color === "primary"
      ? pct >= 80
        ? "bg-emerald-500"
        : pct >= 50
        ? "bg-amber-500"
        : "bg-red-500"
      : "bg-sky-500";

  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-2 rounded-full bg-surface-100 overflow-hidden">
        <div
          className={`h-full rounded-full ${clr} transition-all duration-700 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-foreground/80 w-12 text-right">
        {pct}%
      </span>
    </div>
  );
}

export default function AnalyzePage() {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStage, setCurrentStage] = useState(-1);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAlternatives, setShowAlternatives] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [cameraId, setCameraId] = useState("USER_UPLOAD");
  const [locationField, setLocationField] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (JPG, PNG, GIF, TIFF, WEBP).");
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setCurrentStage(-1);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragOver(false), []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0)
        handleFile(e.target.files[0]);
    },
    [handleFile]
  );

  const analyzeImage = useCallback(async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setCurrentStage(0);

    // Simulate stage progression while we wait for the backend
    const stageTimer = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev < PIPELINE_STAGES.length - 1) return prev + 1;
        return prev;
      });
    }, 400);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("camera_id", cameraId);
      if (locationField.trim()) formData.append("location", locationField.trim());

      const res = await fetch(`${API_BASE}/api/images/analyze`, {
        method: "POST",
        body: formData,
      });

      clearInterval(stageTimer);

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setCurrentStage(PIPELINE_STAGES.length);
      setResult(data.data);
    } catch (err: unknown) {
      clearInterval(stageTimer);
      setError(err instanceof Error ? err.message : "Analysis failed");
      setCurrentStage(-1);
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile, cameraId, locationField]);

  const resetAll = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setCurrentStage(-1);
    setShowAlternatives(false);
    setShowReasoning(false);
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const decisionColor = (d: string) => {
    if (d === "auto_accept") return "text-emerald-400";
    if (d === "human_review") return "text-amber-400";
    return "text-red-400";
  };
  const decisionBg = (d: string) => {
    if (d === "auto_accept") return "bg-emerald-500/15 border-emerald-500/30";
    if (d === "human_review") return "bg-amber-500/15 border-amber-500/30";
    return "bg-red-500/15 border-red-500/30";
  };
  const decisionLabel = (d: string) => {
    if (d === "auto_accept") return "Auto Accepted";
    if (d === "human_review") return "Needs Human Review";
    return "Uncertain";
  };
  const decisionIcon = (d: string) => {
    if (d === "auto_accept") return CheckCircle2;
    if (d === "human_review") return Eye;
    return AlertTriangle;
  };

  // Determine the primary species from classification or decision
  const primarySpecies =
    result?.decision?.species ||
    (result?.detections?.[0]?.classifications?.[0]?.species) ||
    null;
  const primaryConf =
    result?.decision?.confidence ?? 0;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <header className="border-b border-white/10 pb-4">
        <h1 className="text-3xl font-bold tracking-tight text-white mb-1 flex items-center gap-3">
          <Sparkles size={28} className="text-primary-400" />
          Analyze Image
        </h1>
        <p className="text-xs text-foreground/70">
          Upload any camera-trap image to identify species, assess quality, and
          get AI-powered conservation intelligence
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* ====== LEFT COLUMN: Upload + Preview ====== */}
        <div className="space-y-5">
          {/* Upload Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => inputRef.current?.click()}
            className={`relative rounded-xl border-2 border-dashed cursor-pointer transition-all duration-300 ${
              dragOver
                ? "border-primary-400 bg-primary-900/30 scale-[1.01]"
                : selectedFile
                ? "border-primary-500/40 bg-surface-100"
                : "border-white/15 bg-surface-50 hover:border-primary-500/40 hover:bg-surface-100"
            } ${selectedFile ? "p-3" : "p-10"}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleInputChange}
            />

            {previewUrl ? (
              <div className="relative">
                <img
                  src={previewUrl}
                  alt="Selected"
                  className="w-full max-h-[400px] object-contain rounded-lg"
                />
                {/* Bounding Box Overlay */}
                {result?.detections?.map((det, i) => (
                  <div
                    key={i}
                    className="absolute border-2 border-emerald-400 rounded"
                    style={{
                      left: `${det.bbox.x_min * 100}%`,
                      top: `${det.bbox.y_min * 100}%`,
                      width: `${(det.bbox.x_max - det.bbox.x_min) * 100}%`,
                      height: `${(det.bbox.y_max - det.bbox.y_min) * 100}%`,
                    }}
                  >
                    <span className="absolute -top-5 left-0 text-[10px] font-bold bg-emerald-500 text-white px-1.5 py-0.5 rounded">
                      {det.object_type} {Math.round(det.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center">
                <Upload
                  size={48}
                  className="mx-auto text-foreground/30 mb-4"
                />
                <p className="text-base font-medium text-foreground/60 mb-1">
                  Drag & drop a camera-trap image here
                </p>
                <p className="text-xs text-foreground/40">
                  or click to browse · JPG, PNG, GIF, TIFF, WEBP up to 50 MB
                </p>
              </div>
            )}
          </div>

          {/* File info & metadata fields */}
          {selectedFile && (
            <div className="glass-card rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-white truncate max-w-xs">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-foreground/50">
                    {formatBytes(selectedFile.size)} ·{" "}
                    {selectedFile.type.split("/")[1]?.toUpperCase()}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    resetAll();
                  }}
                  className="text-xs text-foreground/50 hover:text-red-400 transition-colors px-2 py-1 rounded hover:bg-red-500/10"
                >
                  Clear
                </button>
              </div>

              {/* Optional metadata */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-foreground/40 mb-1 block">
                    <Camera size={10} className="inline mr-1" />
                    Camera ID
                  </label>
                  <input
                    type="text"
                    value={cameraId}
                    onChange={(e) => setCameraId(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded-lg text-xs bg-surface-100 border border-white/10 text-white placeholder:text-foreground/30"
                    placeholder="CAM001"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-wider text-foreground/40 mb-1 block">
                    <MapPin size={10} className="inline mr-1" />
                    Location
                  </label>
                  <input
                    type="text"
                    value={locationField}
                    onChange={(e) => setLocationField(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded-lg text-xs bg-surface-100 border border-white/10 text-white placeholder:text-foreground/30"
                    placeholder="Waterhole Alpha"
                  />
                </div>
              </div>

              {/* Analyze Button */}
              <button
                onClick={analyzeImage}
                disabled={isAnalyzing}
                className={`w-full py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all duration-300 ${
                  isAnalyzing
                    ? "bg-surface-200 text-foreground/50 cursor-wait"
                    : "bg-primary-600 hover:bg-primary-500 text-white shadow-lg shadow-primary-900/40 hover:shadow-primary-700/50 active:scale-[0.98]"
                }`}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={18} className="animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    Run AI Pipeline
                  </>
                )}
              </button>
            </div>
          )}

          {/* Pipeline Progress */}
          {(isAnalyzing || currentStage >= 0) && (
            <div className="glass-card rounded-xl p-4">
              <h3 className="text-xs uppercase tracking-wider text-foreground/40 mb-3">
                Pipeline Stages
              </h3>
              <div className="space-y-2">
                {PIPELINE_STAGES.map((stage, idx) => {
                  const Icon = stage.icon;
                  const isComplete = currentStage > idx;
                  const isCurrent = currentStage === idx;
                  const isPending = currentStage < idx;

                  return (
                    <div
                      key={stage.key}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-300 ${
                        isCurrent
                          ? "bg-primary-900/40 border border-primary-500/30"
                          : isComplete
                          ? "bg-surface-100/50"
                          : "opacity-40"
                      }`}
                    >
                      {isComplete ? (
                        <CheckCircle2
                          size={16}
                          className="text-emerald-400 shrink-0"
                        />
                      ) : isCurrent ? (
                        <Loader2
                          size={16}
                          className="text-primary-400 animate-spin shrink-0"
                        />
                      ) : (
                        <Icon
                          size={16}
                          className="text-foreground/30 shrink-0"
                        />
                      )}
                      <span
                        className={`text-xs font-medium ${
                          isComplete
                            ? "text-emerald-400"
                            : isCurrent
                            ? "text-white"
                            : "text-foreground/40"
                        }`}
                      >
                        {stage.label}
                      </span>
                      {isComplete && (
                        <span className="ml-auto text-[10px] text-emerald-500/70">
                          ✓
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
              {result && (
                <div className="mt-3 pt-3 border-t border-white/5 text-center">
                  <span className="text-xs text-emerald-400 font-medium">
                    ✓ Pipeline complete in{" "}
                    {result.pipeline_time_ms.toFixed(0)}ms
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ====== RIGHT COLUMN: Results ====== */}
        <div className="space-y-5">
          {/* Error */}
          {error && (
            <div className="glass-card rounded-xl p-4 border-red-500/30 bg-red-500/10">
              <div className="flex items-center gap-2 text-red-400">
                <XCircle size={18} />
                <span className="text-sm font-medium">Analysis Failed</span>
              </div>
              <p className="text-xs text-red-300/70 mt-1">{error}</p>
            </div>
          )}

          {/* No result yet placeholder */}
          {!result && !error && !isAnalyzing && (
            <div className="glass-card rounded-xl p-12 text-center">
              <ScanEye
                size={56}
                className="mx-auto text-foreground/15 mb-4"
              />
              <p className="text-sm text-foreground/40 font-medium">
                Upload an image and click{" "}
                <span className="text-primary-400">Run AI Pipeline</span> to
                see results
              </p>
              <p className="text-xs text-foreground/25 mt-2">
                The pipeline will run Quality Gate → MegaDetector → SpeciesNet →
                OpenCLIP → SAM2 → Decision Engine
              </p>
            </div>
          )}

          {/* ======= RESULTS ======= */}
          {result && (
            <>
              {/* 1. Primary Species ID Card */}
              {primarySpecies && result.decision && (
                <div
                  className={`rounded-xl p-5 border ${decisionBg(
                    result.decision.decision
                  )}`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <p className="text-[10px] uppercase tracking-wider text-foreground/40 mb-1">
                        Species Identified
                      </p>
                      <h2 className="text-2xl font-bold text-white flex items-center gap-2">
                        {result.decision.is_tiger && (
                          <span className="text-xl">🐯</span>
                        )}
                        {primarySpecies}
                      </h2>
                    </div>
                    {(() => {
                      const DIcon = decisionIcon(result.decision.decision);
                      return (
                        <div
                          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-bold ${decisionBg(
                            result.decision.decision
                          )} ${decisionColor(result.decision.decision)}`}
                        >
                          <DIcon size={14} />
                          {decisionLabel(result.decision.decision)}
                        </div>
                      );
                    })()}
                  </div>

                  {/* Confidence */}
                  <div className="mb-3">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-foreground/60">
                        Overall Confidence
                      </span>
                      <span
                        className={`font-bold ${decisionColor(
                          result.decision.decision
                        )}`}
                      >
                        {Math.round(primaryConf * 100)}%
                      </span>
                    </div>
                    <ConfidenceBar value={primaryConf} />
                  </div>

                  {/* Confidence Level Badge */}
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-foreground/50">Confidence Level:</span>
                    <span
                      className={`px-2 py-0.5 rounded-full font-semibold ${
                        result.decision.confidence_level === "high"
                          ? "bg-emerald-500/20 text-emerald-400"
                          : result.decision.confidence_level === "medium"
                          ? "bg-amber-500/20 text-amber-400"
                          : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {result.decision.confidence_level.toUpperCase()}
                    </span>
                  </div>
                </div>
              )}

              {/* Empty frame / no detections */}
              {result.detections.length === 0 && result.quality.passed && (
                <div className="glass-card rounded-xl p-5 text-center">
                  <p className="text-foreground/60 text-sm">
                    No animals, humans, or vehicles detected.
                  </p>
                  <p className="text-foreground/40 text-xs mt-1">
                    This appears to be an empty frame.
                  </p>
                </div>
              )}

              {/* Quality gate failed */}
              {!result.quality.passed && (
                <div className="glass-card rounded-xl p-5 border border-red-500/20 bg-red-900/10">
                  <div className="flex items-center gap-2 text-red-400 mb-2">
                    <XCircle size={18} />
                    <span className="text-sm font-bold">
                      Quality Gate: REJECTED
                    </span>
                  </div>
                  <p className="text-xs text-red-300/70">
                    Status:{" "}
                    <span className="font-mono uppercase">
                      {result.quality.status}
                    </span>
                    . This image was too low quality for reliable AI analysis.
                  </p>
                </div>
              )}

              {/* 2. Detections Grid */}
              {result.detections.length > 0 && (
                <div className="glass-card rounded-xl p-4">
                  <h3 className="text-xs uppercase tracking-wider text-foreground/40 mb-3 flex items-center gap-2">
                    <ScanEye size={14} />
                    Detections ({result.detections.length})
                  </h3>
                  <div className="space-y-3">
                    {result.detections.map((det, i) => (
                      <div
                        key={i}
                        className="bg-surface-100 rounded-lg p-3 border border-white/5"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-white capitalize flex items-center gap-2">
                            {det.object_type === "animal"
                              ? "🦁"
                              : det.object_type === "human"
                              ? "🚶"
                              : "🚗"}
                            {det.object_type}
                          </span>
                          <span className="text-xs font-mono text-foreground/60">
                            {Math.round(det.confidence * 100)}% conf
                          </span>
                        </div>
                        <ConfidenceBar value={det.confidence} />

                        {/* Classification for this detection */}
                        {det.classifications.map((cls, j) => (
                          <div key={j} className="mt-3 pl-3 border-l-2 border-primary-500/30">
                            <p className="text-xs text-foreground/60">
                              SpeciesNet →{" "}
                              <span className="text-white font-semibold">
                                {cls.species}
                              </span>{" "}
                              ({Math.round(cls.confidence * 100)}%)
                            </p>
                            {cls.alternatives?.length > 0 && (
                              <div className="mt-1">
                                <button
                                  onClick={() =>
                                    setShowAlternatives(!showAlternatives)
                                  }
                                  className="text-[10px] text-primary-400 flex items-center gap-1 hover:underline"
                                >
                                  {showAlternatives
                                    ? "Hide"
                                    : `Show ${cls.alternatives.length}`}{" "}
                                  alternatives
                                  {showAlternatives ? (
                                    <ChevronUp size={10} />
                                  ) : (
                                    <ChevronDown size={10} />
                                  )}
                                </button>
                                {showAlternatives && (
                                  <div className="mt-1 space-y-1">
                                    {cls.alternatives.map(
                                      (
                                        alt: {
                                          species: string;
                                          confidence: number;
                                        },
                                        k: number
                                      ) => (
                                        <div
                                          key={k}
                                          className="flex justify-between text-[11px]"
                                        >
                                          <span className="text-foreground/50">
                                            {alt.species}
                                          </span>
                                          <span className="text-foreground/40 font-mono">
                                            {Math.round(alt.confidence * 100)}%
                                          </span>
                                        </div>
                                      )
                                    )}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 3. OpenCLIP Verification */}
              {result.verification && (
                <div className="glass-card rounded-xl p-4">
                  <h3 className="text-xs uppercase tracking-wider text-foreground/40 mb-3 flex items-center gap-2">
                    <Eye size={14} />
                    OpenCLIP Semantic Verification
                  </h3>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-white font-medium">
                      {result.verification.primary_prediction}
                    </span>
                    <span className="text-xs font-mono text-foreground/60">
                      {Math.round(result.verification.confidence * 100)}%
                      similarity
                    </span>
                  </div>
                  <ConfidenceBar
                    value={result.verification.confidence}
                    color="info"
                  />

                  {/* Agreement indicator */}
                  {primarySpecies && (
                    <div className="mt-2 text-xs">
                      {result.verification.primary_prediction.toLowerCase() ===
                      primarySpecies.toLowerCase() ? (
                        <span className="text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 size={12} /> Models agree
                        </span>
                      ) : (
                        <span className="text-amber-400 flex items-center gap-1">
                          <AlertTriangle size={12} /> Models disagree —
                          SpeciesNet: {primarySpecies}, OpenCLIP:{" "}
                          {result.verification.primary_prediction}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* 4. Quality Metrics */}
              <div className="glass-card rounded-xl p-4">
                <h3 className="text-xs uppercase tracking-wider text-foreground/40 mb-3 flex items-center gap-2">
                  <FileCheck size={14} />
                  Quality Assessment
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    {
                      label: "Status",
                      value: result.quality.status?.toUpperCase(),
                      color: result.quality.passed
                        ? "text-emerald-400"
                        : "text-red-400",
                    },
                    {
                      label: "Quality Score",
                      value:
                        result.quality.score != null
                          ? `${Math.round(result.quality.score * 100)}%`
                          : "—",
                      color: "text-white",
                    },
                    {
                      label: "Blur Score",
                      value:
                        result.quality.blur_score != null
                          ? result.quality.blur_score.toFixed(1)
                          : "—",
                      color: "text-white",
                    },
                    {
                      label: "Brightness",
                      value:
                        result.quality.brightness != null
                          ? result.quality.brightness.toFixed(1)
                          : "—",
                      color: "text-white",
                    },
                    {
                      label: "Contrast",
                      value:
                        result.quality.contrast != null
                          ? result.quality.contrast.toFixed(1)
                          : "—",
                      color: "text-white",
                    },
                    {
                      label: "Dimensions",
                      value:
                        result.image.width && result.image.height
                          ? `${result.image.width}×${result.image.height}`
                          : "—",
                      color: "text-white",
                    },
                  ].map((item, i) => (
                    <div
                      key={i}
                      className="bg-surface-100 rounded-lg px-3 py-2"
                    >
                      <p className="text-[10px] text-foreground/40 uppercase tracking-wider">
                        {item.label}
                      </p>
                      <p className={`text-sm font-semibold ${item.color}`}>
                        {item.value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* 5. Decision Reasoning */}
              {result.decision?.reasoning &&
                result.decision.reasoning.length > 0 && (
                  <div className="glass-card rounded-xl p-4">
                    <button
                      onClick={() => setShowReasoning(!showReasoning)}
                      className="w-full flex items-center justify-between text-xs uppercase tracking-wider text-foreground/40 hover:text-foreground/60 transition-colors"
                    >
                      <span className="flex items-center gap-2">
                        <Brain size={14} />
                        AI Reasoning ({result.decision.reasoning.length} signals)
                      </span>
                      {showReasoning ? (
                        <ChevronUp size={14} />
                      ) : (
                        <ChevronDown size={14} />
                      )}
                    </button>
                    {showReasoning && (
                      <ul className="mt-3 space-y-1.5">
                        {result.decision.reasoning.map((r, i) => (
                          <li
                            key={i}
                            className="text-xs text-foreground/60 flex items-start gap-2"
                          >
                            <span className="text-primary-400 mt-0.5">•</span>
                            {r}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}

              {/* 6. Pipeline Meta */}
              <div className="text-center text-[10px] text-foreground/30 pt-2">
                Image ID: <span className="font-mono">{result.image_id}</span> ·
                Processed in {result.pipeline_time_ms.toFixed(0)}ms
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
