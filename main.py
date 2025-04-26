# main.py
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import json
import cv2
import base64
from scipy.signal import butter, lfilter, periodogram, detrend
from collections import deque
import asyncio

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Signal Processing Functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def calculate_hr(signal, fs=30.0):
    if len(signal) < fs * 3:  # Need at least 3 seconds of data
        return None
    
    # Apply bandpass filter
    filtered = bandpass_filter(detrend(signal), fs=fs)
    
    # Calculate power spectrum
    f, Pxx = periodogram(filtered, fs)
    
    # Find peak in the valid heart rate range (45-180 BPM)
    valid = (f >= 0.75) & (f <= 3.0)  # 0.75-3.0 Hz = 45-180 BPM
    
    if np.sum(valid) == 0 or np.max(Pxx[valid]) < 1e-5:
        return None
    
    peak_freq = f[valid][np.argmax(Pxx[valid])]
    bpm = peak_freq * 60
    
    return bpm

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Initialize data structures
    fs = 30.0  # Target sampling rate
    window = deque(maxlen=int(fs * 10))  # 10 second rolling window
    green_signal = []
    timestamps = []
    hr_values = []
    hr_times = []
    
    last_hr_calc_time = 0
    start_time = None
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            
            # Initialize start time if first frame
            if start_time is None:
                start_time = 0
                
            # Parse the JSON data
            frame_data = json.loads(data)
            
            # Extract the base64 image and timestamp
            image_base64 = frame_data.get("image").split(",")[1]
            client_timestamp = frame_data.get("timestamp", 0) / 1000.0  # Convert to seconds
            
            # If this is the first frame, reset the start time
            if not timestamps:
                start_time = client_timestamp
                
            current_time = client_timestamp - start_time
            
            # Decode the image
            image_bytes = base64.b64decode(image_base64)
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            display_frame = frame.copy()
            face_detected = False
            hr = None
            
            if len(faces) > 0:
                face_detected = True
                # Get largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract forehead region (upper 1/3 of face)
                forehead_height = int(h * 0.3)
                forehead_y = max(0, y + int(h * 0.1))
                forehead = frame[forehead_y:forehead_y + forehead_height, x:x + w]
                
                # Draw rectangle around forehead
                cv2.rectangle(display_frame, (x, forehead_y), (x + w, forehead_y + forehead_height), (0, 0, 255), 2)
                
                if forehead.size > 0:
                    # Extract green channel average
                    green = np.mean(forehead[:, :, 1])
                    
                    # Store data
                    timestamps.append(current_time)
                    green_signal.append(green)
                    window.append(green)
                    
                    # Calculate heart rate every second
                    if (current_time - last_hr_calc_time >= 1.0 and len(window) >= fs * 3):
                        last_hr_calc_time = current_time
                        
                        hr = calculate_hr(np.array(window), fs)
                        
                        if hr is not None and 45 <= hr <= 180:
                            hr_values.append(hr)
                            hr_times.append(current_time)
                            
                            # Add to display frame
                            cv2.putText(display_frame, f"HR: {hr:.1f} BPM", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        else:
                            cv2.putText(display_frame, "HR: Calculating...", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Add time to display
            cv2.putText(display_frame, f"Time: {current_time:.1f}s", 
                      (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if not face_detected:
                cv2.putText(display_frame, "No Face Detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Encode the frame with the overlays to send back to client
            _, buffer = cv2.imencode('.jpg', display_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            
            # Send data back to client
            await websocket.send_json({
                "processed_image": f"data:image/jpeg;base64,{encoded_frame}",
                "face_detected": face_detected,
                "heart_rate": hr if hr is not None else None,
                "current_time": current_time
            })
            
            # Brief pause to avoid overwhelming the websocket
            await asyncio.sleep(0.01)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")