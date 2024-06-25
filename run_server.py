import os
import time
import cv2
import mediapipe as mp
import math
import numpy as np
import torch
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)

# Declaring FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.3, min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Model Initialization
model_path = os.path.join(os.path.dirname(__file__), "models", "clf_lstm_jit6.pth")
model = torch.jit.load(model_path)
model.eval()

# Default normalization values (adjust these based on typical calibration results)
ears_norm = [0.3, 0.02]
mars_norm = [0.5, 0.02]
pucs_norm = [0.6, 0.02]
moes_norm = [1.7, 0.1]

right_eye = [
    [33, 133],
    [160, 144],
    [159, 145],
    [158, 153],
]  # right eye landmark positions
left_eye = [
    [263, 362],
    [387, 373],
    [386, 374],
    [385, 380],
]  # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]  # mouth landmark coordinates
states = ["not-drowsy", "drowsy"]


def distance(p1, p2):
    """Calculate distance between two points"""
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5


def eye_aspect_ratio(landmarks, eye):
    """Calculate the ratio of the eye length to eye width."""
    try:
        N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
        N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
        N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
        D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
        return (N1 + N2 + N3) / (3 * D)
    except Exception as e:
        print(f"Error calculating eye aspect ratio: {e}")
        return None


def eye_feature(landmarks):
    """Calculate the eye feature as the average of the eye aspect ratio for the two eyes"""
    left_ear = eye_aspect_ratio(landmarks, left_eye)
    right_ear = eye_aspect_ratio(landmarks, right_eye)
    if left_ear is not None and right_ear is not None:
        return (left_ear + right_ear) / 2
    return None


def mouth_feature(landmarks):
    """Calculate mouth feature as the ratio of the mouth length to mouth width"""
    try:
        N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
        N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
        N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
        D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
        return (N1 + N2 + N3) / (3 * D)
    except Exception as e:
        print(f"Error calculating mouth feature: {e}")
        return None


def pupil_circularity(landmarks, eye):
    """Calculate pupil circularity feature."""
    try:
        perimeter = (
            distance(landmarks[eye[0][0]], landmarks[eye[1][0]])
            + distance(landmarks[eye[1][0]], landmarks[eye[2][0]])
            + distance(landmarks[eye[2][0]], landmarks[eye[3][0]])
            + distance(landmarks[eye[3][0]], landmarks[eye[0][1]])
            + distance(landmarks[eye[0][1]], landmarks[eye[3][1]])
            + distance(landmarks[eye[3][1]], landmarks[eye[2][1]])
            + distance(landmarks[eye[2][1]], landmarks[eye[1][1]])
            + distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
        )
        area = math.pi * (
            (distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2
        )
        return (4 * math.pi * area) / (perimeter**2)
    except Exception as e:
        print(f"Error calculating pupil circularity: {e}")
        return None


def pupil_feature(landmarks):
    """Calculate the pupil feature as the average of the pupil circularity for the two eyes"""
    left_puc = pupil_circularity(landmarks, left_eye)
    right_puc = pupil_circularity(landmarks, right_eye)
    if left_puc is not None and right_puc is not None:
        return (left_puc + right_puc) / 2
    return None


def run_face_mp(image):
    """Get face landmarks using the FaceMesh MediaPipe model.
    Calculate facial features using the landmarks."""
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append([data_point.x, data_point.y, data_point.z])
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                landmark_drawing_spec=drawing_spec,
            )

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear if ear else None

        if ear is None or mar is None or puc is None or moe is None:
            print(
                f"Error: Feature extraction failed with landmarks: {landmarks_positions}"
            )
    else:
        ear, mar, puc, moe = None, None, None, None

    return ear, mar, puc, moe, image


def get_classification(features):
    """Perform classification over the facial features.
    :param features: List of normalized facial features
    :return: Alert / Drowsy state prediction
    """
    # Padding the input to match the model's expected input shape
    padded_features = np.zeros((20, 4))
    padded_features[:1] = features  # Place the single frame at the beginning

    model_input = torch.FloatTensor(np.array([padded_features]))  # Adjust input shape
    preds = torch.sigmoid(model(model_input)).gt(0.5).int().data.numpy()
    return int(preds[0][0])


def classify_image(image_path):
    image = cv2.imread(image_path)
    ear, mar, puc, moe, _ = run_face_mp(image)

    if ear is None or mar is None or puc is None or moe is None:
        return "Error: Failed to extract features from the image"

    features = [ear, mar, puc, moe]
    normalized_features = [
        (features[0] - ears_norm[0]) / ears_norm[1],
        (features[1] - mars_norm[0]) / mars_norm[1],
        (features[2] - pucs_norm[0]) / pucs_norm[1],
        (features[3] - moes_norm[0]) / moes_norm[1],
    ]

    label = get_classification(normalized_features)
    return states[label]


@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_name = request.form.get("image_name", "uploaded_image.jpg")
    image = Image.open(BytesIO(file.read()))
    image_path = os.path.join("uploads", image_name)

    os.makedirs("uploads", exist_ok=True)
    image.save(image_path)

    result = classify_image(image_path)
    os.remove(image_path)

    return jsonify({"driver_state": result}), 200


@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = base64.b64decode(data["image"])
    image = Image.open(BytesIO(image_data))
    image_path = os.path.join("uploads", "stream_image.jpg")

    os.makedirs("uploads", exist_ok=True)
    image.save(image_path)

    result = classify_image(image_path)
    os.remove(image_path)

    return jsonify({"driver_state": result}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
