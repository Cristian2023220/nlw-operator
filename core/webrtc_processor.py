import av
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time

class GestureVideoProcessor:
    def __init__(self):
        # Setup models
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
        
        # Load custom models
        self.MP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_recognizer.task")
        self.CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.joblib")
        self.ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.joblib")
        
        self.clf = joblib.load(self.CUSTOM_MODEL_PATH)
        self.label_encoder = joblib.load(self.ENCODER_PATH)
        
        # Setup mediapipe
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        # MediaPipe for landmarks
        # We use IMAGE mode so we don't have to send tracking timestamps manually per frame
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.MP_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(self.options)
        
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        
        # Process the image
        img = cv2.flip(image, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        recognition_result = self.recognizer.recognize(mp_image)
        
        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # 1. Desenha os landmarks
                self.mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                # 2. Prepara dados para o modelo customizado
                # Lidar com o caso se não houver handedness
                if recognition_result.handedness and len(recognition_result.handedness) > i:
                    hand_label = recognition_result.handedness[i][0].category_name
                else:
                    hand_label = "Unknown"
                    
                handedness_val = 0 if hand_label == 'Left' else 1
                
                landmarks_array = [handedness_val]
                for lm in hand_landmarks:
                    landmarks_array.extend([lm.x, lm.y, lm.z])
                
                features = np.array(landmarks_array).reshape(1, -1)
                
                # Predição do modelo customizado
                prediction_idx = self.clf.predict(features)[0]
                prediction_prob = np.max(self.clf.predict_proba(features))
                gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                # 3. Exibe o resultado
                color = (0, 255, 0) # Verde
                display_text = f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
                cv2.putText(img, display_text, (20, 50 + (i * 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")
