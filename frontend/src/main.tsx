import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { useDropzone } from 'react-dropzone';
import { AnimatePresence, motion } from 'framer-motion';
import {
  FiArrowLeft,
  FiCheck,
  FiColumns,
  FiDownload,
  FiImage,
  FiMaximize,
  FiRefreshCw,
  FiRotateCcw,
  FiRotateCw,
  FiSliders,
  FiUpload,
  FiZoomIn,
  FiZoomOut,
} from 'react-icons/fi';
import './styles.css';

const rawApiUrl = import.meta.env.VITE_API_URL;
const API_BASE = (rawApiUrl && rawApiUrl.trim() !== '' ? rawApiUrl : '/api').replace(/\/+$/, '');

type Point = [number, number];

type UploadData = {
  session_id: string;
  image_url: string;
  width: number;
  height: number;
};

const ENHANCEMENT_MODES = [
  ['black_white', 'B&W Clean'],
] as const;

function formatApiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function extractErrorMessage(response: Response): Promise<string> {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    const body = await response.json().catch(() => null);
    if (body?.detail) return body.detail;
  }
  if (response.status === 404) {
    return 'Session expired or image resource not found. Please upload again.';
  }
  return `Server request failed with status ${response.status}.`;
}

function createCombinedSignal(signal1?: AbortSignal | null, signal2?: AbortSignal | null): AbortSignal {
  const controller = new AbortController();

  const onAbort = () => {
    if (!controller.signal.aborted) {
      controller.abort();
    }
  };

  if (signal1) {
    if (signal1.aborted) {
      controller.abort();
      return controller.signal;
    }
    signal1.addEventListener('abort', onAbort, { once: true });
  }

  if (signal2) {
    if (signal2.aborted) {
      controller.abort();
      return controller.signal;
    }
    signal2.addEventListener('abort', onAbort, { once: true });
  }

  return controller.signal;
}

async function apiFetch(
  url: string,
  options: RequestInit & { timeoutMs?: number; retries?: number } = {}
): Promise<Response> {
  const { timeoutMs = 60000, retries = 2, signal, ...fetchOptions } = options;

  let attempt = 0;
  while (attempt <= retries) {
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), timeoutMs);
    const combinedSignal = createCombinedSignal(signal, timeoutController.signal);

    try {
      const response = await fetch(url, { ...fetchOptions, signal: combinedSignal });
      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error(await extractErrorMessage(response));
      }
      return response;
    } catch (err) {
      clearTimeout(timeoutId);
      if (signal?.aborted) {
        throw new Error('Request cancelled.');
      }
      if (timeoutController.signal.aborted) {
        throw new Error('Server request timed out. Please try again.');
      }
      if (attempt < retries && err instanceof Error && err.name !== 'AbortError') {
        attempt++;
        await new Promise((res) => setTimeout(res, 800 * attempt));
        continue;
      }
      throw err;
    }
  }
  throw new Error('Network request failed after retries.');
}

function preloadImage(url: string): Promise<void> {
  return new Promise((resolve) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve();
    img.onerror = () => resolve();
    img.src = url;
  });
}

function App() {
  const [step, setStep] = useState<number>(0);
  const [upload, setUpload] = useState<UploadData | undefined>();
  const [corners, setCorners] = useState<Point[]>([]);
  const [cropUrl, setCropUrl] = useState<string>('');
  const [finalUrl, setFinalUrl] = useState<string>('');
  const [mode, setMode] = useState<string>('black_white');
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const enhanceAbortRef = useRef<AbortController | null>(null);

  const handleUpload = async (file: File) => {
    setBusy(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await apiFetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data: UploadData = await uploadRes.json();
      const url = formatApiUrl(data.image_url);
      preloadImage(url);

      const detectRes = await apiFetch(`${API_BASE}/detect?session_id=${data.session_id}`, {
        method: 'POST',
      });

      const detectData = await detectRes.json();
      setUpload(data);
      setCorners(detectData.corners);
      setStep(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to upload or process this image.');
    } finally {
      setBusy(false);
    }
  };

  const handleAutoDetect = async () => {
    if (!upload) return;
    setBusy(true);
    setError('');
    try {
      const detectRes = await apiFetch(`${API_BASE}/detect?session_id=${upload.session_id}`, {
        method: 'POST',
      });
      const detectData = await detectRes.json();
      setCorners(detectData.corners);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Automatic corner detection was unavailable.');
    } finally {
      setBusy(false);
    }
  };

  const handleCrop = async () => {
    if (!upload) return;
    setBusy(true);
    setError('');
    try {
      const cropRes = await apiFetch(`${API_BASE}/crop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: upload.session_id,
          corners,
        }),
      });

      const data = await cropRes.json();
      const url = formatApiUrl(data.image_url);
      preloadImage(url);
      setCropUrl(url);
      setStep(2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Please keep all four corners inside the image.');
    } finally {
      setBusy(false);
    }
  };

  const handleEnhance = async (selectedMode = mode) => {
    if (!upload) return;

    if (enhanceAbortRef.current) {
      enhanceAbortRef.current.abort();
    }
    const controller = new AbortController();
    enhanceAbortRef.current = controller;

    setBusy(true);
    setError('');
    try {
      const enhanceRes = await apiFetch(`${API_BASE}/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: upload.session_id,
          mode: selectedMode,
        }),
        signal: controller.signal,
      });

      const data = await enhanceRes.json();
      const url = formatApiUrl(data.image_url);
      setFinalUrl(url);
      preloadImage(url);
    } catch (err) {
      if (err instanceof Error && err.message === 'Request cancelled.') return;
      setError(err instanceof Error ? err.message : 'Could not apply selected enhancement.');
    } finally {
      if (enhanceAbortRef.current === controller) {
        setBusy(false);
      }
    }
  };

  useEffect(() => {
    if (step === 3 && upload) {
      handleEnhance();
    }
  }, [step]);

  const restartScanner = () => {
    if (enhanceAbortRef.current) {
      enhanceAbortRef.current.abort();
    }
    setStep(0);
    setUpload(undefined);
    setCropUrl('');
    setFinalUrl('');
    setCorners([]);
    setMode('black_white');
    setError('');
  };

  return (
    <main>
      <header>
        <div className="brand">
          <span>▣</span> Paperly DocScanner <em>v1.0</em>
        </div>
        <div className="secure">
          <div className="secure-badge">
            <div className="secure-dot" /> OPENCV ENGINE READY
          </div>
        </div>
      </header>

      <ProgressIndicator currentStep={step} />

      <AnimatePresence mode="wait">
        <motion.section
          key={step}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.22 }}
        >
          {step === 0 && <HomeStep onUpload={handleUpload} busy={busy} />}
          {step === 1 && upload && (
            <CropStep
              image={formatApiUrl(upload.image_url)}
              corners={corners}
              imageWidth={upload.width}
              imageHeight={upload.height}
              onChange={setCorners}
              onNext={handleCrop}
              onBack={restartScanner}
              onAuto={handleAutoDetect}
              busy={busy}
            />
          )}
          {step === 2 && upload && (
            <CompareStep
              original={formatApiUrl(upload.image_url)}
              corrected={cropUrl}
              onBack={() => setStep(1)}
              onNext={() => setStep(3)}
            />
          )}
          {step === 3 && (
            <EnhanceStep
              image={finalUrl}
              mode={mode}
              setMode={setMode}
              onEnhance={handleEnhance}
              busy={busy}
              onBack={() => setStep(2)}
              onNext={() => setStep(4)}
            />
          )}
          {step === 4 && upload && (
            <FinalStep image={finalUrl} session={upload.session_id} restart={restartScanner} />
          )}
        </motion.section>
      </AnimatePresence>

      {error && (
        <div className="toast">
          <span>{error}</span>
          <button onClick={() => setError('')} title="Dismiss">
            ×
          </button>
        </div>
      )}
    </main>
  );
}

function ProgressIndicator({ currentStep }: { currentStep: number }) {
  const steps = ['Upload', 'Crop', 'Compare', 'Enhance', 'Done'];
  return (
    <div className="progress" role="progressbar" aria-valuenow={currentStep + 1} aria-valuemin={1} aria-valuemax={5} aria-label={`Step ${currentStep + 1} of 5: ${steps[currentStep]}`}>
      {steps.map((label, idx) => {
        const isActive = idx <= currentStep;
        return (
          <div className={isActive ? 'active' : ''} key={label} aria-current={idx === currentStep ? 'step' : undefined}>
            <i>{idx < currentStep ? <FiCheck /> : idx + 1}</i>
            <span>{label}</span>
          </div>
        );
      })}
    </div>
  );
}

function HomeStep({ onUpload, busy }: { onUpload: (file: File) => void; busy: boolean }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'],
    },
    multiple: false,
    onDrop: (files) => files[0] && onUpload(files[0]),
  });

  return (
    <div className="hero">
      <div>
        <div className="eyebrow-badge">⚡ AI & OPENCV POWERED</div>
        <h1>
          Turn photos into
          <br />
          <em>crisp documents.</em>
        </h1>
        <p className="lede">
          Automatic boundary detection, perspective correction, and smart contrast enhancements powered by high precision computer vision.
        </p>
      </div>
      <div {...getRootProps()} className={`drop ${isDragActive ? 'hover' : ''}`}>
        <input {...getInputProps()} />
        {busy ? (
          <div className="spinner-box">
            <div className="spinner" />
            <p>Analyzing document...</p>
          </div>
        ) : (
          <>
            <div className="upload-icon">
              <FiUpload />
            </div>
            <h2>{isDragActive ? 'Drop file here' : 'Upload your document'}</h2>
            <p>Drag & drop your file or click to browse</p>
            <button className="primary" type="button">
              <FiImage /> Choose Document
            </button>
            <div className="format-pills">
              <span className="format-pill">JPG</span>
              <span className="format-pill">PNG</span>
              <span className="format-pill">WEBP</span>
              <span className="format-pill">BMP</span>
              <span className="format-pill">TIFF</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function CropStep({
  image,
  corners,
  imageWidth,
  imageHeight,
  onChange,
  onNext,
  onBack,
  onAuto,
  busy,
}: {
  image: string;
  corners: Point[];
  imageWidth: number;
  imageHeight: number;
  onChange: (points: Point[]) => void;
  onNext: () => void;
  onBack: () => void;
  onAuto: () => void;
  busy: boolean;
}) {
  return (
    <div className="workspace">
      <aside>
        <button className="back" onClick={onBack} type="button">
          <FiArrowLeft /> Choose Different Image
        </button>
        <p className="eyebrow">STEP 2 OF 5</p>
        <h1>Adjust Crop Corners</h1>
        <p>
          Drag corner markers to paper edges. Drag background to pan, or use mouse wheel to zoom.
        </p>

        <div className="button-group">
          <button className="secondary" onClick={onAuto} disabled={busy} type="button">
            <FiRefreshCw /> Auto Detect
          </button>
          <button className="primary wide" disabled={busy} onClick={onNext} type="button">
            Perspective Crop <FiCheck />
          </button>
        </div>
      </aside>

      <div className="preview-card">
        <InteractiveImageViewer src={image} alt="Document crop source" showRotateControls={true}>
          <CropOverlay
            points={corners}
            naturalWidth={imageWidth}
            naturalHeight={imageHeight}
            setPoints={onChange}
          />
        </InteractiveImageViewer>
      </div>
    </div>
  );
}

type InteractiveImageViewerProps = {
  src: string;
  alt: string;
  children?: React.ReactNode;
  showRotateControls?: boolean;
};

function InteractiveImageViewer({
  src,
  alt,
  children,
  showRotateControls = true,
}: InteractiveImageViewerProps) {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [rotation, setRotation] = useState<number>(0);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const panRef = useRef(pan);
  panRef.current = pan;

  useEffect(() => {
    if (zoom <= 1.0) {
      setPan({ x: 0, y: 0 });
    }
  }, [zoom]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.max(0.5, Math.min(4.0, parseFloat((z * factor).toFixed(2)))));
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.pointerType === 'mouse') return;

    const startX = e.clientX - panRef.current.x;
    const startY = e.clientY - panRef.current.y;
    setIsDragging(true);

    const onPointerMove = (moveEvent: PointerEvent) => {
      setPan({
        x: moveEvent.clientX - startX,
        y: moveEvent.clientY - startY,
      });
    };

    const onPointerUp = () => {
      setIsDragging(false);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
    setRotation(0);
  };

  return (
    <div
      className={`interactive-viewer ${isDragging ? 'panning' : ''}`}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onDoubleClick={resetView}
    >
      <div
        className="viewer-viewport"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom}) rotate(${rotation}deg)`,
          transition: isDragging ? 'none' : 'transform 0.15s ease-out',
        }}
      >
        <div className="image-relative-container">
          <img src={src} alt={alt} draggable={false} decoding="async" />
          {children}
        </div>
      </div>

      <div className="viewer-toolbar">
        <button
          onClick={() => setZoom((z) => Math.max(0.5, parseFloat((z - 0.25).toFixed(2))))}
          title="Zoom Out"
          type="button"
        >
          <FiZoomOut />
        </button>
        <span className="zoom-value-label">{Math.round(zoom * 100)}%</span>
        <button
          onClick={() => setZoom((z) => Math.min(4.0, parseFloat((z + 0.25).toFixed(2))))}
          title="Zoom In"
          type="button"
        >
          <FiZoomIn />
        </button>
        {showRotateControls && (
          <>
            <button
              onClick={() => setRotation((r) => (r + 90) % 360)}
              title="Rotate 90°"
              type="button"
            >
              <FiRotateCw />
            </button>
            <button
              onClick={() => setRotation((r) => (r - 90 + 360) % 360)}
              title="Rotate -90°"
              type="button"
            >
              <FiRotateCcw />
            </button>
          </>
        )}
        <button onClick={resetView} title="Fit Entire Image" type="button">
          <FiMaximize /> Fit Image
        </button>
        <button onClick={() => setZoom(1.0)} title="100% Actual Size" type="button">
          100%
        </button>
      </div>
    </div>
  );
}

function CropOverlay({
  points,
  naturalWidth,
  naturalHeight,
  setPoints,
}: {
  points: Point[];
  naturalWidth: number;
  naturalHeight: number;
  setPoints: (points: Point[]) => void;
}) {
  const [activeCorner, setActiveCorner] = useState<number>(-1);

  const handleCornerPointerDown = (index: number, e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setActiveCorner(index);

    const container = (e.currentTarget as HTMLElement).closest('.image-relative-container');
    if (!container) return;

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      let relX = (moveEvent.clientX - rect.left) / rect.width;
      let relY = (moveEvent.clientY - rect.top) / rect.height;

      relX = Math.max(0, Math.min(1, relX));
      relY = Math.max(0, Math.min(1, relY));

      const realX = Math.round(relX * naturalWidth);
      const realY = Math.round(relY * naturalHeight);

      const newPoints = [...points];
      newPoints[index] = [realX, realY];
      setPoints(newPoints);
    };

    const handlePointerUp = () => {
      setActiveCorner(-1);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
  };

  const normalizedPoints = points.map(([x, y]) => [
    x / (naturalWidth || 1),
    y / (naturalHeight || 1),
  ]);

  const polygonPointsStr = normalizedPoints.map((p) => `${p[0]},${p[1]}`).join(' ');

  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      className="crop-svg-overlay"
      style={{ pointerEvents: 'none' }}
    >
      <polygon
        points={polygonPointsStr}
        className="crop-polygon"
        style={{ pointerEvents: 'none' }}
      />
      {normalizedPoints.map((p, i) => (
        <g key={i} style={{ pointerEvents: 'none' }}>
          {/* Transparent enlarged touch hit-target */}
          <circle
            cx={p[0]}
            cy={p[1]}
            r={0.045}
            fill="transparent"
            style={{ cursor: 'grab', pointerEvents: 'auto' }}
            onPointerDown={(e) => handleCornerPointerDown(i, e)}
            aria-label={`Crop Corner ${i + 1}`}
          />
          {/* Visible handle circle */}
          <circle
            cx={p[0]}
            cy={p[1]}
            r={0.022}
            className={`crop-handle ${activeCorner === i ? 'dragging' : ''}`}
            style={{ pointerEvents: 'auto' }}
            onPointerDown={(e) => handleCornerPointerDown(i, e)}
          />
        </g>
      ))}
    </svg>
  );
}

function CompareStep({
  original,
  corrected,
  onBack,
  onNext,
}: {
  original: string;
  corrected: string;
  onBack: () => void;
  onNext: () => void;
}) {
  const [viewMode, setViewMode] = useState<'slider' | 'side'>('slider');
  const [splitPos, setSplitPos] = useState<number>(50);
  const [isSliding, setIsSliding] = useState<boolean>(false);
  const sliderRef = useRef<HTMLDivElement>(null);

  const handleSliderMove = (clientX: number) => {
    if (sliderRef.current) {
      const rect = sliderRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
      setSplitPos(pct);
    }
  };

  return (
    <div className="center-page">
      <p className="eyebrow">STEP 3 OF 5</p>
      <h1>Perspective Correction Result</h1>
      <p className="lede">
        Your document has been flattened and rectified. Slide to compare original photo vs scan.
      </p>

      <div className="compare-toggle-bar">
        <button
          className={viewMode === 'slider' ? 'active' : ''}
          onClick={() => setViewMode('slider')}
          type="button"
        >
          <FiSliders /> Split Slider
        </button>
        <button
          className={viewMode === 'side' ? 'active' : ''}
          onClick={() => setViewMode('side')}
          type="button"
        >
          <FiColumns /> Side by Side
        </button>
      </div>

      {viewMode === 'slider' ? (
        <div
          ref={sliderRef}
          className="split-comparison"
          onPointerDown={(e) => {
            setIsSliding(true);
            handleSliderMove(e.clientX);
          }}
          onPointerMove={(e) => isSliding && handleSliderMove(e.clientX)}
          onPointerUp={() => setIsSliding(false)}
          onPointerLeave={() => setIsSliding(false)}
        >
          <img className="split-bg" src={original} alt="Original input photo" />
          <div
            className="split-fg-wrapper"
            style={{ clipPath: `inset(0 ${100 - splitPos}% 0 0)` }}
          >
            <img className="split-fg" src={corrected} alt="Rectified perspective scan" />
          </div>

          <div className="split-divider" style={{ left: `${splitPos}%` }}>
            <div className="split-handle">↔</div>
          </div>

          <span className="badge badge-left">Original</span>
          <span className="badge badge-right">Rectified</span>
        </div>
      ) : (
        <div className="comparison">
          <figure>
            <img src={original} alt="Original input photo" />
            <figcaption>Original Photo</figcaption>
          </figure>
          <div className="arrow">→</div>
          <figure>
            <img src={corrected} alt="Rectified perspective scan" />
            <figcaption>Rectified Scan</figcaption>
          </figure>
        </div>
      )}

      <div className="actions">
        <button className="secondary" onClick={onBack} type="button">
          Adjust Crop Corners
        </button>
        <button className="primary" onClick={onNext} type="button">
          Choose Image Filter <FiCheck />
        </button>
      </div>
    </div>
  );
}

function EnhanceStep({
  image,
  mode,
  setMode,
  onEnhance,
  busy,
  onBack,
  onNext,
}: {
  image: string;
  mode: string;
  setMode: (m: string) => void;
  onEnhance: (m: string) => void;
  busy: boolean;
  onBack: () => void;
  onNext: () => void;
}) {
  return (
    <div className="workspace enhance">
      <aside>
        <button className="back" onClick={onBack} type="button">
          <FiArrowLeft /> Back to Comparison
        </button>
        <p className="eyebrow">STEP 4 OF 5</p>
        <h1>Enhance & Clean</h1>
        <p>Select an enhancement filter. Use viewer controls to inspect full document details.</p>
        <div className="modes">
          {ENHANCEMENT_MODES.map(([id, label]) => (
            <button
              className={mode === id ? 'selected' : ''}
              key={id}
              disabled={busy}
              onClick={() => {
                setMode(id);
                onEnhance(id);
              }}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
        <button className="primary wide" onClick={onNext} disabled={!image || busy} type="button">
          Final Review <FiCheck />
        </button>
      </aside>

      <div className="preview-card">
        {busy ? (
          <div className="spinner-box">
            <div className="spinner" />
            <p>Applying filter...</p>
          </div>
        ) : (
          <InteractiveImageViewer src={image} alt="Enhanced document preview" showRotateControls={true} />
        )}
      </div>
    </div>
  );
}

function FinalStep({
  image,
  session,
  restart,
}: {
  image: string;
  session: string;
  restart: () => void;
}) {
  return (
    <div className="center-page final">
      <div className="eyebrow-badge">✓ SCAN COMPLETED</div>
      <h1>Your Document is Ready</h1>
      <p className="lede">Crisp, straightened, high contrast scan ready for instant export.</p>
      <img className="final-image" src={image} alt="Final processed scan result" />
      <div className="actions">
        <a className="primary" href={`${API_BASE}/download?session_id=${session}&format=png`}>
          <FiDownload /> Download PNG
        </a>
        <a className="secondary" href={`${API_BASE}/download?session_id=${session}&format=jpg`}>
          <FiDownload /> Download JPG
        </a>
        <button className="text-button" onClick={restart} type="button">
          Scan Another Document
        </button>
      </div>
    </div>
  );
}

const container = document.getElementById('root');
if (container) {
  createRoot(container).render(<App />);
}
