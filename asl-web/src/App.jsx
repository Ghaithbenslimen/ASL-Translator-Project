import { useEffect, useRef, useState, useCallback } from 'react'
import * as ort from 'onnxruntime-web'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import './App.css'

const CLASS_NAMES = [
  'A','B','C','D','E','F','G','H','I','J','K','L','M',
  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
  'del','nothing','space',
]
const IMG_SIZE = 128
const CONF_THRESHOLD = 0.65
const DEL_CONF_THRESHOLD = 0.45
const STABLE_WINDOW = 12
const STABLE_CONFIRM = 8
const RESET_FRAMES = 4
const SPACE_SECONDS = 2.5
const PAD = 0.15

class LetterStabilizer {
  constructor() {
    this.window = []
    this.current = null
    this.emptyStreak = 0
  }
  update(letter, conf) {
    const threshold = letter === 'del' ? DEL_CONF_THRESHOLD : CONF_THRESHOLD
    if (letter == null || letter === 'nothing' || conf < threshold) {
      this.emptyStreak++
      if (this.emptyStreak >= RESET_FRAMES) {
        this.current = null
        this.window = []
      }
      return null
    }
    this.emptyStreak = 0
    this.window.push(letter)
    if (this.window.length > STABLE_WINDOW) this.window.shift()
    const counts = {}
    for (const l of this.window) counts[l] = (counts[l] || 0) + 1
    const [top, votes] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
    if (votes >= STABLE_CONFIRM && top !== this.current) {
      this.current = top
      this.window = []
      return top
    }
    return null
  }
  reset() {
    this.window = []
    this.current = null
    this.emptyStreak = 0
  }
}

function cropToTensor(canvas, srcX, srcY, srcW, srcH) {
  const off = document.createElement('canvas')
  off.width = IMG_SIZE
  off.height = IMG_SIZE
  const ctx = off.getContext('2d')
  ctx.drawImage(canvas, srcX, srcY, srcW, srcH, 0, 0, IMG_SIZE, IMG_SIZE)
  const imgData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data
  const size = IMG_SIZE * IMG_SIZE
  const float32 = new Float32Array(3 * size)
  for (let i = 0; i < size; i++) {
    const r = imgData[i * 4 + 0] / 255
    const g = imgData[i * 4 + 1] / 255
    const b = imgData[i * 4 + 2] / 255
    float32[i] = (r - 0.5) / 0.5
    float32[size + i] = (g - 0.5) / 0.5
    float32[2 * size + i] = (b - 0.5) / 0.5
  }
  return new ort.Tensor('float32', float32, [1, 3, IMG_SIZE, IMG_SIZE])
}

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const sessionRef = useRef(null)
  const landmarkerRef = useRef(null)
  const stabilizerRef = useRef(new LetterStabilizer())
  const rafRef = useRef(null)
  const lastLetterTimeRef = useRef(Date.now())
  const copyTimerRef = useRef(null)

  const [ready, setReady] = useState(false)
  const [text, setText] = useState('')
  const [predicted, setPredicted] = useState(null)
  const [confidence, setConfidence] = useState(0)
  const [handDetected, setHandDetected] = useState(false)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  // -------- Load ONNX model + HandLandmarker --------
  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/'
        const session = await ort.InferenceSession.create('/asl_net.onnx', {
          executionProviders: ['wasm'],
        })
        if (cancelled) return
        sessionRef.current = session

        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        )
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.6,
          minTrackingConfidence: 0.6,
        })
        if (cancelled) return
        landmarkerRef.current = landmarker
        setReady(true)
      } catch (e) {
        console.error(e)
        setError(String(e))
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  // -------- Webcam (Strict-Mode-safe) --------
  useEffect(() => {
    let stream = null
    let isMounted = true

    async function startCam() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        })

        if (!isMounted) {
          stream.getTracks().forEach(track => track.stop())
          return
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          const playPromise = videoRef.current.play()
          if (playPromise !== undefined) {
            playPromise.catch(err => {
              console.log('Video play interrupted (safe in Strict Mode):', err.name)
            })
          }
        }
      } catch (e) {
        if (isMounted) {
          setError('Camera access denied: ' + e.message)
        }
      }
    }

    startCam()

    return () => {
      isMounted = false
      if (stream) {
        stream.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // -------- Main inference loop --------
  const loop = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const session = sessionRef.current
    const landmarker = landmarkerRef.current
    if (!video || !canvas || !session || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loop)
      return
    }

    const t0 = performance.now()
    const ctx = canvas.getContext('2d')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.drawImage(video, 0, 0)

    const result = landmarker.detectForVideo(video, t0)

    if (result.landmarks && result.landmarks.length > 0) {
      setHandDetected(true)
      const lms = result.landmarks[0]
      const xs = lms.map(p => p.x)
      const ys = lms.map(p => p.y)
      const w = canvas.width
      const h = canvas.height

      let x1 = Math.max(0, (Math.min(...xs) - PAD) * w)
      let y1 = Math.max(0, (Math.min(...ys) - PAD) * h)
      let x2 = Math.min(w, (Math.max(...xs) + PAD) * w)
      let y2 = Math.min(h, (Math.max(...ys) + PAD) * h)

      const side = Math.max(x2 - x1, y2 - y1)
      const cx = (x1 + x2) / 2
      const cy = (y1 + y2) / 2
      x1 = Math.max(0, cx - side / 2)
      y1 = Math.max(0, cy - side / 2)
      x2 = Math.min(w, x1 + side)
      y2 = Math.min(h, y1 + side)

      const tensor = cropToTensor(canvas, x1, y1, x2 - x1, y2 - y1)
      session.run({ input: tensor }).then(out => {
        const logits = out.logits.data
        const max = Math.max(...logits)
        const exps = logits.map(v => Math.exp(v - max))
        const sum = exps.reduce((a, b) => a + b, 0)
        const probs = exps.map(v => v / sum)
        let bestIdx = 0
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[bestIdx]) bestIdx = i
        }
        const letter = CLASS_NAMES[bestIdx]
        const conf = probs[bestIdx]

        setPredicted(letter)
        setConfidence(conf)

        const newLetter = stabilizerRef.current.update(letter, conf)
        if (newLetter) {
          lastLetterTimeRef.current = Date.now()
          if (newLetter === 'del') {
            setText(t => t.slice(0, -1))
          } else if (newLetter === 'space') {
            setText(t => (t && !t.endsWith(' ') ? t + ' ' : t))
          } else if (newLetter !== 'nothing') {
            setText(t => t + newLetter)
          }
        }
      })
    } else {
      setHandDetected(false)
      stabilizerRef.current.update(null, 0)
    }

    if (Date.now() - lastLetterTimeRef.current > SPACE_SECONDS * 1000) {
      setText(t => (t && !t.endsWith(' ') ? t + ' ' : t))
      lastLetterTimeRef.current = Date.now()
    }

    const dt = performance.now() - t0
    setFps(Math.round(1000 / Math.max(1, dt)))
    rafRef.current = requestAnimationFrame(loop)
  }, [])

  useEffect(() => {
    if (!ready) return
    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [ready, loop])

  // -------- Keyboard shortcuts --------
  useEffect(() => {
    function onKey(e) {
      if (e.key === ' ') {
        e.preventDefault()
        setText(t => (t && !t.endsWith(' ') ? t + ' ' : t))
      } else if (e.key === 'b' || e.key === 'B') {
        setText(t => t.slice(0, -1))
      } else if (e.key === 'c' || e.key === 'C') {
        setText('')
        stabilizerRef.current.reset()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // -------- Copy with feedback --------
  const handleCopy = useCallback(async () => {
    if (!text || !navigator.clipboard) return
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      clearTimeout(copyTimerRef.current)
      copyTimerRef.current = setTimeout(() => setCopied(false), 1600)
    } catch (e) {
      console.error('Copy failed:', e)
    }
  }, [text])

  useEffect(() => () => clearTimeout(copyTimerRef.current), [])

  const confidenceTier =
    confidence >= 0.8 ? 'is-high' : confidence >= 0.6 ? 'is-mid' : 'is-low'

  return (
    <div className="app">
      <header>
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">A</div>
          <div className="brand-text">
            <h1>ASL Translator</h1>
            <p className="brand-sub">Real-time fingerspelling · runs entirely in your browser</p>
          </div>
        </div>
        <p className={`status-pill ${error ? 'is-error' : ready ? 'is-ready' : 'is-loading'}`}>
          {error ? 'Error' : ready ? `Live · ${fps} FPS` : 'Loading model…'}
        </p>
      </header>

      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}

      <div className={`stage ${ready ? 'is-ready' : 'is-loading'}${handDetected ? ' is-live' : ''}`}>
        <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
        <canvas ref={canvasRef} className="video-canvas" />
        <div className="overlay">
          <div className={`status ${handDetected ? 'is-on' : 'is-off'}`}>
            {handDetected ? 'Hand detected' : 'No hand'}
          </div>
          {predicted && (
            <div
              className={`prediction ${confidenceTier}`}
              style={{ '--conf': confidence }}
            >
              {predicted} ({(confidence * 100).toFixed(0)}%)
            </div>
          )}
        </div>
      </div>

      <div className="output" data-count={text.length}>
        <label>Translation</label>
        <div className="text-banner" aria-live="polite">
          {text.replace(/ /g, '·') || (
            <span className="placeholder">start signing…</span>
          )}
        </div>
        <div className="controls">
          <button type="button" className="primary-action" onClick={() => setText('')}>Clear (C)</button>
          <button type="button" onClick={() => setText(t => t.slice(0, -1))}>Backspace (B)</button>
          <button type="button" onClick={() => setText(t => (t && !t.endsWith(' ') ? t + ' ' : t))}>
            Space
          </button>
          <button
            type="button"
            className={`copy-action${copied ? ' is-copied' : ''}`}
            onClick={handleCopy}
            disabled={!text}
            aria-live="polite"
          >
            {copied ? 'Copied!' : 'Copy text'}
          </button>
        </div>
        <div className="hint-row">
          <p className="hint">
            Hold a sign steady to commit it. Pause or lower your hand for a word break.
          </p>
          <span className="shortcut-note">C clear <b>•</b> B delete <b>•</b> Space break</span>
        </div>
      </div>
    </div>
  )
}