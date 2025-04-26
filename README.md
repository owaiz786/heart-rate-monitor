# Heart Rate Monitor Web Application

This is a web-based heart rate monitoring application built with FastAPI. It uses your webcam to detect your face and estimate your heart rate in real-time.

## Features

- Real-time heart rate monitoring using webcam
- Face detection and tracking
- Live signal visualization
- Heart rate trend tracking
- Responsive web interface

## How It Works

1. The application uses your webcam to capture video frames
2. It detects your face and identifies the forehead region
3. The green channel of the forehead region is analyzed for subtle color changes
4. These color changes correlate with blood flow, which can be used to estimate heart rate
5. Signal processing techniques extract the heart rate from this data

## Project Structure

```
├── main.py               # FastAPI application
├── templates/
│   └── index.html        # Web interface template
├── static/               # Static files (if needed)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Local Development Setup

1. Clone this repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   uvicorn main:app --reload
   ```
5. Open your browser and navigate to `http://localhost:8000`

## Deploying to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Use the following settings:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Click "Create Web Service"

### Important Notes for Render Deployment

- Make sure to add `opencv-python-headless` to your requirements.txt instead of `opencv-python` for deployment
- Ensure your Render service has sufficient memory allocation (at least 512MB)
- The application requires HTTPS for webcam access in most browsers when deployed

## Usage Tips

- Position your face clearly in the frame
- Ensure you have good lighting (natural light works best)
- Try to remain still during measurement
- Allow 10-15 seconds for initial heart rate calculation
- For best results, avoid excessive movement

## Limitations

- This application is for demonstration purposes only and not intended for medical use
- Results may vary based on lighting conditions, skin tone, and camera quality
- The heart rate estimation algorithm is based on photoplethysmography (PPG) principles but is simplified for web use

## License

This project is licensed under the MIT License - see the LICENSE file for details.