import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

class GestureProcessor:
    def __init__(self):
        # Base setup
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
        
        # Paths to models
        self.MP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_recognizer.task")
        self.CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.joblib")
        self.ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder.joblib")
        
        # Load sklearn custom models
        self.clf = joblib.load(self.CUSTOM_MODEL_PATH)
        self.label_encoder = joblib.load(self.ENCODER_PATH)
        
        # Setup MediaPipe hand tracking (IMAGE mode for frame-by-frame)
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
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

    def process_frame(self, image: np.ndarray, draw_landmarks: bool = True):
        """
        Process a single image frame for hand landmarks + custom gesture classification.
        Args:
            image: BGR image from opencv/webrtc.
            draw_landmarks: Whether to draw points over the image or just return data.
        Returns:
            processed_img: The BGR image mapped with visual indicators if draw_landmarks=True.
            labels: List of dicts with detected hand info.
            gesture_image: None (or you can return cropped hand image if needed, keeping it None for simplicity here).
        """
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        recognition_result = self.recognizer.recognize(mp_image)
        
        processed_img = image.copy()
        labels = []
        
        if recognition_result.hand_landmarks:
            for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # Optionally Draw Landmarks
                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        processed_img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Determine Handedness
                if recognition_result.handedness and len(recognition_result.handedness) > i:
                    hand_label = recognition_result.handedness[i][0].category_name
                else:
                    hand_label = "Unknown"
                    
                handedness_val = 0 if hand_label == 'Left' else 1
                
                # Format for Joblib custom model
                landmarks_array = [handedness_val]
                for lm in hand_landmarks:
                    landmarks_array.extend([lm.x, lm.y, lm.z])
                
                features = np.array(landmarks_array).reshape(1, -1)
                
                # Predict
                prediction_idx = self.clf.predict(features)[0]
                prediction_prob = np.max(self.clf.predict_proba(features))
                gesture_name = self.label_encoder.inverse_transform([prediction_idx])[0]

                # Append to labels for websocket/webrtc feedback
                labels.append({
                    "hand": hand_label,
                    "gesture": gesture_name,
                    "confidence": float(prediction_prob)
                })

                if draw_landmarks:
                    color = (0, 255, 0)
                    display_text = f"{hand_label}: {gesture_name} ({prediction_prob:.2f})"
                    cv2.putText(processed_img, display_text, (20, 50 + (i * 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                
        return processed_img, labels, None
