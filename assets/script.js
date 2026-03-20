function initApp() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    let ws;

    navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    }).then(stream => {
        video.srcObject = stream;
        video.play().catch(e => console.error("Play error:", e));
        
        const onMetadata = () => {
            canvas.width = tempCanvas.width = video.videoWidth;
            canvas.height = tempCanvas.height = video.videoHeight;
            initWS();
        };

        if (video.readyState >= 1) { // HAVE_METADATA is 1
            onMetadata();
        } else {
            video.addEventListener('loadedmetadata', onMetadata, { once: true });
        }
    }).catch(err => {
        console.error("Camera access Error:", err);
        alert("Erro ao acessar a câmera: " + err.message + "\nPor favor, permita o acesso à câmera no seu navegador.");
    });

    function initWS() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0);

                // Update FPS
                const fpsCounter = document.getElementById('fps-counter');
                if (fpsCounter && data.fps !== undefined) {
                    fpsCounter.textContent = `FPS: ${data.fps}`;
                }

                // Update labels
                const labelContainer = document.getElementById('gesture-container');
                if (labelContainer) {
                    if (data.labels.length === 0) {
                        labelContainer.innerHTML = '<div class="label-item"><span class="name">Sem gestos detectados</span></div>';
                    } else {
                        labelContainer.innerHTML = data.labels.map(l =>
                            `<div class="label-item">
                                <span class="name">${l.hand}: ${l.gesture}</span>
                                <span class="prob">${(l.confidence * 100).toFixed(1)}%</span>
                             </div>`
                        ).join('');
                    }
                }

                const gestureImg = document.getElementById('gesture-image');
                if (gestureImg) {
                    if (data.gesture_image) {
                        gestureImg.src = `/assets/images/gestures/${data.gesture_image}`;
                        gestureImg.style.display = 'block';
                        gestureImg.classList.add('active-gesture');
                    } else {
                        gestureImg.style.display = 'none';
                        gestureImg.classList.remove('active-gesture');
                    }
                }

                sendFrame();
            };
            img.src = data.image;
        };
        ws.onopen = sendFrame;
        ws.onclose = () => setTimeout(initWS, 1000);
    }

    const qualitySlider = document.getElementById('quality-slider');
    const qualityValue = document.getElementById('quality-value');
    const drawLandmarksCb = document.getElementById('draw-landmarks-cb');
    let currentQuality = 0.6;

    if (qualitySlider && qualityValue) {
        qualitySlider.oninput = function () {
            currentQuality = parseFloat(this.value);
            qualityValue.textContent = Math.round(currentQuality * 100) + '%';
        };
    }

    function sendFrame() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            tempCtx.drawImage(video, 0, 0);
            const drawLandmarks = drawLandmarksCb ? drawLandmarksCb.checked : true;
            ws.send(JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', currentQuality),
                draw_landmarks: drawLandmarks
            }));
        }
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}