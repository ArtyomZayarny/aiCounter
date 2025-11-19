# AI Counter - Utility Meter Reading with OCR

Mobile application for apartment owners to automatically capture and recognize utility meter readings using OCR technology.

## Project Structure

```
aiCounter-app/
├── client/          # Flutter mobile app
├── server/          # Python backend (FastAPI)
└── docs/            # Documentation
```

## Prerequisites

### For Flutter Client

- Flutter SDK (latest stable)
- Android Studio / Xcode for mobile development
- Dart SDK (included with Flutter)

### For Python Backend

- Python 3.8 or higher
- Tesseract OCR installed on your system
- Virtual environment (recommended)

## Setup Instructions

### 1. Flutter Client Setup

```bash
cd client
flutter pub get
flutter run
```

### 2. Python Backend Setup

#### Install Tesseract OCR

**macOS:**

```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

#### Setup Python Environment

```bash
cd server
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Run Backend Server

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs

## Development

### Verify Tesseract Installation

```bash
tesseract --version
```

### Test Flutter Build

```bash
cd client
flutter build apk  # For Android
flutter build ios  # For iOS
```

## Project Status

This project is in active development. See `.taskmaster/tasks/tasks.json` for current task list.
