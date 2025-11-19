# AI Counter Backend Server

Python FastAPI backend server for utility meter reading with OCR.

## Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- **GET** `/health`
- Returns server status

### Upload Image
- **POST** `/api/upload-image`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Image file (required)
  - `utility_type`: Type of meter - "gas", "water", or "electricity" (required)
- **Response:** JSON with file information

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Test health endpoint:
```bash
curl http://localhost:8000/health
```

Test image upload:
```bash
curl -X POST "http://localhost:8000/api/upload-image" \
  -F "file=@path/to/image.jpg" \
  -F "utility_type=gas"
```

## Project Structure

```
server/
├── app/
│   ├── api/
│   │   └── routes.py      # API endpoints
│   ├── ocr/               # OCR processing (future)
│   ├── processing/        # Image processing (future)
│   └── models/            # Data models (future)
├── uploads/               # Temporary file storage
├── main.py                # Application entry point
└── requirements.txt       # Python dependencies
```

