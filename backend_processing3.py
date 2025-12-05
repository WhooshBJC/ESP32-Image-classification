# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:48:40 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 21:49:58 2025

@author: User
"""

import asyncio
import nest_asyncio
import websockets
import cv2
import numpy as np
import traceback
from multiprocessing import Manager, Pool, Queue
import torch
#import ssl

import torchvision
from torchvision.transforms import transforms
from PIL import Image

from ResNetArchitecture2 import ResNet50

#ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
#ssl_context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

device = torch.device('cuda')
net = ResNet50(img_channel=3, num_classes=12).to(device)
net.load_state_dict(torch.load("resnet_weights.pth"))
net.eval()
#{'battery': 0, 'biological': 1, 'brown-glass': 2, 'cardboard': 3, 'clothes': 4, 'green-glass': 5, 'metal': 6, 'paper': 7, 'plastic': 8, 'shoes': 9, 'trash': 10, 'white-glass': 11}
class_map= {'battery': 0, 'biological': 1, 'brown-glass': 2, 'cardboard': 3, 'clothes': 4, 'green-glass': 5, 'metal': 6, 'paper': 7, 'plastic': 8, 'shoes': 9, 'trash': 10, 'white-glass': 11}
corrected_class_map = {v:k for k,v in class_map.items()}

data_dir = "/PythonProject/data/GarbageClassification/garbage_classification"
transform_for_stat = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.6581,0.6162,0.5856], [0.2117,0.2115,0.2157])])
dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform_for_stat)

esp32_clients = set()
browser_clients = set()
nest_asyncio.apply()

def box_and_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    
    for c in contours:
        if cv2.contourArea(c) > 500 and cv2.contourArea(c) < 1500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, "detected", (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0),2)
            detected = True
            
    return frame, detected

    

def decode_frame(message):
    img_array = np.frombuffer(message, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return  frame

async def handle_connection(websocket):
    esp32_clients.add(websocket)
    print(f"[ESP32] Connected: {websocket.remote_address}")
    #global latest_frame
    try:
        async for message in websocket:
            print(len(message))
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, process_frame, message)
            print (result[0])
            if len(message) > 2000:
                #loop = asyncio.get_event_loop()
                #result = await loop.run_in_executor(pool, process_frame, message)
                if result is not None:
                    predicted_class, frame_to_send = result
                else:
                    print('no result')
                
                if predicted_class is not None:
                    #await websocket.send(predicted_class)
                    try:
                      frame_queue.put_nowait(frame_to_send)
                    except:
                        try:
                           frame_queue.get_nowait()
                           frame_queue.put_nowait(frame_to_send)
                        except:
                            pass

    except websockets.exceptions.ConnectionClosed:
        print(f"[ESP32] Disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"[ESP32] Error: {e}")
        traceback.print_exc()
    finally:
        esp32_clients.discard(websocket)

placeholder = cv2.imread('placeholder.jpg')

def process_frame(message):
    try:
        frame = decode_frame(message)
        
        if frame is None:
            print("Corrupt frame, skipping")
            return
                
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        print(f"Frame decoded: {frame.shape}")
        
        if box_and_detect(frame):
            with torch.no_grad():
                input_tensor = transform_for_stat(image).unsqueeze(0).to(device)
                outputs = net(input_tensor)
                _, predicted = torch.max(outputs, 1)
            
            
            prediction = corrected_class_map[predicted.item()]
            print(f"Predicted class: {prediction}")
            return prediction, frame
        
        else:
            print("No objecet detected, skipping")
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()

async def stream_frames(websocket):
    browser_clients.add(websocket)
    print(f"[Browser] Connected: {websocket.remote_address}")
    latest_frame = placeholder
    frame_count = 0
    no_frame_count = 0
    
    try:
        while True:
            got_frame = False
            try:
                frame_to_send = frame_queue.get_nowait()
                latest_frame = frame_to_send
                got_frame = True
                no_frame_count = 0
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"[Browser] Sent {frame_count} real frames")
                
            except:
                frame_to_send = latest_frame
                no_frame_count += 1
                if no_frame_count == 30:
                    print(f"[Browser] WARNING: No new frames for 1 second (queue size: {frame_queue.qsize()})")
                    
            _, buffer = cv2.imencode('.jpg', frame_to_send)
            await websocket.send(buffer.tobytes())
            await asyncio.sleep(1.0/30)
            
    except websockets.exceptions.ConnectionClosed:
        print(f"[Browser] Disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"[Browser] Unexpected error: {e}")
    finally:
        browser_clients.discard(websocket)
    
async def status_monitor():
    pre_esp32_count = 0
    prev_browser_count = 0
    
    while True:
        current_esp32_count = len(esp32_clients)
        current_browser_count = len(browser_clients)
        
        if current_esp32_count != pre_esp32_count or current_browser_count != prev_browser_count:
            print(f"[status] ESP32: [{len(esp32_clients)}] | Browser: {[len(browser_clients)]}")
            prev_browser_count = current_browser_count
            pre_esp32_count = current_esp32_count
            
        await asyncio.sleep(1)
        
async def main():
    global pool
    
    print("=" * 60)
    print("WebSocket Video Streaming Server Started")
    print("=" * 60)
    print("ESP32 camera input:  ws://<IP>:8008")
    print("Browser video output: ws://<IP>:8000")
    print("=" * 60)
    
    asyncio.create_task(status_monitor())
    
    async with websockets.serve(handle_connection, '0.0.0.0', 8008, max_size=None, ping_interval=20, ping_timeout=60):
        async with websockets.serve(stream_frames, '0.0.0.0', 8000, max_size=None, ping_interval=20, ping_timeout=60):
           try:
               await asyncio.Future()
           except KeyboardInterrupt:
               print ('Cancelling task')
           finally:
               
               for client in list(esp32_clients):
                   await client.close()
               for client in list(browser_clients):
                   await client.close()
                 
               print("[SHUTDOWN] Server stopped")
    
if __name__ == "__main__":
    from multiprocessing import freeze_support
    from concurrent.futures import ProcessPoolExecutor
    freeze_support()
    
    manager = Manager()
    frame_queue = manager.Queue(maxsize=5)
    
    pool = ProcessPoolExecutor(max_workers=6)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Program terminated")
    finally:
        if pool:
            pool.shutdown(wait=True)