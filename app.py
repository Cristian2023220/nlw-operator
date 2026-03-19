import json
import time
from fasthtml.common import *
from core.processor import GestureProcessor
from core.utils import decode_image, encode_image
from starlette.responses import FileResponse

app, rt = fast_app(hdrs=(
    Link(rel='stylesheet', href='/assets/style.css'),
    Script(src='/assets/script.js'),
))
processor = GestureProcessor()

class FPSTracker:
    def __init__(self):
        self.prev_time = time.time()
    def update(self):
        curr = time.time()
        fps = 1 / (curr - self.prev_time) if curr > self.prev_time else 0
        self.prev_time = curr
        return int(fps)

fps_tracker = FPSTracker()

@rt('/assets/{fname:path}')
def serve_assets(fname: str):
    return FileResponse(f'assets/{fname}')

@rt("/")
def get():
    return Title("Rockit Vision — AI Hand Gesture Recognition"), Main(
        Header(
            H1("Rockit Vision"),
            P("Intelligent Hand Gesture Recognition System", cls="subtitle")
        ),
        Div(
            Div(
                Video(id="video", autoplay=True, playsinline=True, style="display:none"),
                Canvas(id="canvas"),
                Div("FPS: 0", id="fps-counter", cls="fps-badge"),
                cls="vision-card"
            ),
            Div(
                Div(
                    H3("Image Quality Control"),
                    Div(
                        Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.05", value="0.6"),
                        Span("60%", id="quality-value"),
                        cls="quality-control"
                    ),
                    H3("Settings"),
                    Div(
                        Label(Input(type="checkbox", id="draw-landmarks-cb", checked=True), " Draw Hand Landmarks"),
                        cls="settings-control"
                    ),
                    H3("Live Feed Data"),
                    Div(id="gesture-container"),
                    cls="info-card"
                ),
                Div(
                    H3("Detected Gesture"),
                    Div(Img(id="gesture-image"), cls="gesture-preview-box"),
                    cls="info-card"
                ),
                cls="sidebar-info"
            ),
            cls="main-content"
        ),
        cls="app-container"
    )

@app.ws("/ws")
async def ws(msg: str, send):
    try:
        data = json.loads(msg)
        img = decode_image(data.get("image", ""))
        
        if img is not None:
            processed_img, labels, _ = processor.process_frame(img, data.get("draw_landmarks", True))
            fps = fps_tracker.update()
            
            gesture_img_name = None
            if len(labels) == 2:
                g1 = labels[0].get("gesture", "").lower()
                g2 = labels[1].get("gesture", "").lower()
                if g1 and g1 == g2:
                    gesture_img_name = f"{g1}.png"
                    
            await send(json.dumps({
                "image": encode_image(processed_img),
                "labels": labels,
                "gesture_image": gesture_img_name,
                "fps": fps
            }))
    except Exception as e:
        print(f"WS error: {e}")

if __name__ == '__main__':
    serve()
