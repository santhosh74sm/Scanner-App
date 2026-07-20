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

const API_BASE = '/api';

type Point = [number, number];

type UploadData = {
  session_id: string;
  image_url: string;
  width: number;
  height: number;
};

const ENHANCEMENT_MODES = [
  ['auto', 'Auto Enhance'],
  ['magic_color', 'Magic Color'],
  ['black_white', 'B&W Clean'],
  ['grayscale', 'Grayscale'],
  ['high_contrast', 'High Contrast'],
  ['color', 'Color Boost'],
  ['original', 'Original'],
] as const;

function formatApiUrl(path: string): string {
  return `${API_BASE}${path}?v=${Date.now()}`;
}

async function extractErrorMessage(response: Response): Promise<string> {
  const body = await response.json().catch(() => null);
  return body?.detail || `Request failed with status ${response.status}.`;
}

function App() {
  const [step, setStep] = useState<number>(0);
  const [upload, setUpload] = useState<UploadData | undefined>();
  const [corners, setCorners] = useState<Point[]>([]);
  const [cropUrl, setCropUrl] = useState<string>('');
  const [finalUrl, setFinalUrl] = useState<string>('');
  const [mode, setMode] = useState<string>('auto');
  const [busy, setBusy] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleUpload = async (file: File) => {
    setBusy(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) {
        throw new Error(await extractErrorMessage(uploadRes));
      }

      const data: UploadData = await uploadRes.json();

      const detectRes = await fetch(`${API_BASE}/detect?session_id=${data.session_id}`, {
        method: 'POST',
      });

      if (!detectRes.ok) {
        throw new Error(await extractErrorMessage(detectRes));
      }

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
      const detectRes = await fetch(`${API_BASE}/detect?session_id=${upload.session_id}`, {
        method: 'POST',
      });
      if (!detectRes.ok) {
        throw new Error(await extractErrorMessage(detectRes));
      }
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
      const cropRes = await fetch(`${API_BASE}/crop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: upload.session_id,
          corners,
        }),
      });

      if (!cropRes.ok) {
        throw new Error(await extractErrorMessage(cropRes));
      }

      const data = await cropRes.json();
      setCropUrl(formatApiUrl(data.image_url));
      setStep(2);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Please keep all four corners inside the image.');
    } finally {
      setBusy(false);
    }
  };

  const handleEnhance = async (selectedMode = mode) => {
    if (!upload) return;
    setBusy(true);
    setError('');
    try {
      const enhanceRes = await fetch(`${API_BASE}/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: upload.session_id,
          mode: selectedMode,
        }),
      });

      if (!enhanceRes.ok) {
        throw new Error(await extractErrorMessage(enhanceRes));
      }

      const data = await enhanceRes.json();
      setFinalUrl(formatApiUrl(data.image_url));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not apply selected enhancement.');
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (step === 3 && upload) {
      handleEnhance();
    }
  }, [step]);

  const restartScanner = () => {
    setStep(0);
    setUpload(undefined);
    setCropUrl('');
    setFinalUrl('');
    setCorners([]);
    setMode('auto');
    setError('');
  };

  return (
    <main>
      <header>
        <div className="brand">
          <span>▣</span> Paperly DocScanner
        </div>
        <div className="secure">OpenCV Document Scanner Engine</div>
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
    <div className="progress">
      {steps.map((label, idx) => {
        const isActive = idx <= currentStep;
        return (
          <div className={isActive ? 'active' : ''} key={label}>
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
        <p className="eyebrow">HIGH PRECISION DOCUMENT SCANNER</p>
        <h1>
          Turn photos into
          <br />
          <em>crisp documents.</em>
        </h1>
        <p className="lede">
          Automatic boundary detection, perspective correction, and smart contrast enhancements powered by OpenCV.
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
            <h2>{isDragActive ? 'Drop file here' : 'Drop your document here'}</h2>
            <p>Supports JPG, PNG, WEBP, BMP, and TIFF</p>
            <button className="primary" type="button">
              <FiImage /> Browse Computer
            </button>
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
  const [zoom, setZoom] = useState<number>(1.0);
  const [rotation, setRotation] = useState<number>(0);

  const handleReset = () => {
    setZoom(1.0);
    setRotation(0);
  };

  return (
    <div className="workspace">
      <aside>
        <button className="back" onClick={onBack} type="button">
          <FiArrowLeft /> Choose Different Image
        </button>
        <p className="eyebrow">STEP 2 OF 5</p>
        <h1>Adjust Crop Corners</h1>
        <p>
          Drag corner markers to paper edges. When zoomed in, drag the document background to pan.
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

      <div className="canvas-card">
        <CropCanvas
          image={image}
          points={corners}
          naturalWidth={imageWidth}
          naturalHeight={imageHeight}
          zoom={zoom}
          rotation={rotation}
          setPoints={onChange}
        />

        <div className="canvas-tools">
          <button
            onClick={() => setZoom((z) => Math.max(0.5, parseFloat((z - 0.25).toFixed(2))))}
            title="Zoom Out"
            type="button"
          >
            <FiZoomOut />
          </button>
          <span className="zoom-value-label">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoom((z) => Math.min(5.0, parseFloat((z + 0.25).toFixed(2))))}
            title="Zoom In"
            type="button"
          >
            <FiZoomIn />
          </button>
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
          <button onClick={handleReset} title="Reset View" type="button">
            <FiMaximize />
          </button>
        </div>
      </div>
    </div>
  );
}

function CropCanvas({
  image,
  points,
  naturalWidth,
  naturalHeight,
  zoom,
  rotation,
  setPoints,
}: {
  image: string;
  points: Point[];
  naturalWidth: number;
  naturalHeight: number;
  zoom: number;
  rotation: number;
  setPoints: (points: Point[]) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [activeCorner, setActiveCorner] = useState<number>(-1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Reset pan when zoom resets to 1.0 (Fit View)
  useEffect(() => {
    if (zoom <= 1.0) {
      setPan({ x: 0, y: 0 });
    }
  }, [zoom]);

  const handleCornerPointerDown = (index: number, e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setActiveCorner(index);
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (activeCorner >= 0) return;
    if (zoom > 1.0) {
      setIsPanning(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (isPanning && zoom > 1.0) {
      const newPanX = e.clientX - dragStart.x;
      const newPanY = e.clientY - dragStart.y;

      // Boundary constraints for smooth clamping
      const maxPanX = (zoom - 1.0) * 350;
      const maxPanY = (zoom - 1.0) * 350;
      const clampedX = Math.max(-maxPanX, Math.min(maxPanX, newPanX));
      const clampedY = Math.max(-maxPanY, Math.min(maxPanY, newPanY));

      setPan({ x: clampedX, y: clampedY });
      return;
    }

    if (activeCorner >= 0 && imgRef.current) {
      const rect = imgRef.current.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      let relX = (e.clientX - rect.left) / rect.width;
      let relY = (e.clientY - rect.top) / rect.height;

      relX = Math.max(0, Math.min(1, relX));
      relY = Math.max(0, Math.min(1, relY));

      const realX = relX * naturalWidth;
      const realY = relY * naturalHeight;

      const newPoints = [...points];
      newPoints[activeCorner] = [Math.round(realX), Math.round(realY)];
      setPoints(newPoints);
    }
  };

  const handlePointerUp = () => {
    setActiveCorner(-1);
    setIsPanning(false);
  };

  const normalizedPoints = points.map(([x, y]) => [
    x / (naturalWidth || 1),
    y / (naturalHeight || 1),
  ]);

  const polygonPointsStr = normalizedPoints.map((p) => `${p[0]},${p[1]}`).join(' ');

  return (
    <div
      className={`crop-canvas-wrapper ${zoom > 1.0 ? 'zoom-pannable' : ''} ${isPanning ? 'panning' : ''}`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      <div
        className="transform-viewport"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom}) rotate(${rotation}deg)`,
          transition: activeCorner >= 0 || isPanning ? 'none' : 'transform 0.15s ease-out',
        }}
      >
        <div className="image-relative-container">
          <img
            ref={imgRef}
            src={image}
            alt="Document crop source"
            draggable={false}
          />
          <svg
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            className="crop-svg-overlay"
          >
            <polygon points={polygonPointsStr} className="crop-polygon" />
            {normalizedPoints.map((p, i) => (
              <circle
                key={i}
                cx={p[0]}
                cy={p[1]}
                r={0.022 / Math.sqrt(zoom)}
                className={`crop-handle ${activeCorner === i ? 'dragging' : ''}`}
                onPointerDown={(e) => handleCornerPointerDown(i, e)}
              />
            ))}
          </svg>
        </div>
      </div>
    </div>
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
          <ImageViewer src={image} alt="Enhanced document preview" />
        )}
      </div>
    </div>
  );
}

function ImageViewer({ src, alt }: { src: string; alt: string }) {
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.max(0.5, Math.min(4.0, parseFloat((z * factor).toFixed(2)))));
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handlePointerUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div
      className="interactive-viewer"
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      onDoubleClick={resetView}
    >
      <div
        className="viewer-viewport"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transition: isDragging ? 'none' : 'transform 0.15s ease-out',
        }}
      >
        <img src={src} alt={alt} draggable={false} />
      </div>

      <div className="viewer-toolbar">
        <button onClick={() => setZoom((z) => Math.max(0.5, parseFloat((z - 0.25).toFixed(2))))} title="Zoom Out" type="button">
          <FiZoomOut />
        </button>
        <span>{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom((z) => Math.min(4.0, parseFloat((z + 0.25).toFixed(2))))} title="Zoom In" type="button">
          <FiZoomIn />
        </button>
        <button onClick={resetView} title="Fit Entire Image" type="button">
          <FiMaximize /> Fit Image
        </button>
        <button onClick={() => { setZoom(1.5); setPan({ x: 0, y: 0 }); }} title="100% Actual Size" type="button">
          100%
        </button>
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
      <p className="eyebrow">SCAN COMPLETED</p>
      <h1>Your Document is Ready</h1>
      <p className="lede">Crisp, straightened, high contrast scan ready for export.</p>
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
