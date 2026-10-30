# 🤟 ASL Fingerspelling Translator

A real-time, browser-based American Sign Language (ASL) fingerspelling translator that runs entirely client-side. The application uses your webcam to detect hand landmarks, crops the hand, and classifies the signed letter (A–Z, `space`, `del`) using a custom ONNX model.

## Key Features

- **Real-time Hand Tracking:** Uses MediaPipe `HandLandmarker` (GPU delegate) to locate 21 hand landmarks and isolate the hand.
- **On-Device Inference:** A custom CNN, exported to **ONNX** (`asl_net.onnx`), runs via `onnxruntime-web` (WASM) directly in the browser.
- **Smart Stabilization:** A sliding-window algorithm filters noisy predictions, requiring consistent recognition across multiple frames before emitting a letter.
- **Professional UI:** A dense, tool-like interface inspired by professional creative software (OBS, Raycast), featuring a live preview of the model's input crop, a confidence bar, and a running text output.

## Dataset

The model was trained on the **ASL Alphabet** dataset.

> **Dataset Link:** https://www.kaggle.com/datasets/grassknoted/asl-alphabet

The dataset contains ~87,000 images of hand signs covering the 26 letters (A–Z) plus `space` and `del`. It is **not** included in this repository due to its size (~1 GB).

## Tech Stack

- **Frontend:** React, Vite
- **Hand Detection:** MediaPipe Tasks Vision (`@mediapipe/tasks-vision`)
- **Model Inference:** ONNX Runtime Web (`onnxruntime-web`)
- **Icons:** Lucide React

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- A webcam
- A modern browser (Chrome or Edge recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ghaithbenslimen/ASL-Translator-Project.git
   cd ASL-Translator-Project