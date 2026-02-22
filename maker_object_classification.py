import cv2
from transformers import pipeline
from ultralytics import YOLO
from PIL import Image
from TTS import speak_label

# Load models
checkpoint = "openai/clip-vit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

labels = ["can", "pencil", "notebook", "phone", "calculator"]
labels = [f'This is a photo of {label}' for label in labels]

cap = cv2.VideoCapture(0) #Webcam
lastLabel = None
lastScore = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed. Exiting...")
        break

    # CLIP pipeline to label object
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(rgb)
    predictions = detector(image, labels)
    
    best = predictions[0]
    label = best["label"]
    score = best["score"]
    frame = cv2.putText(frame, f"{label}: {score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Read current object aloud if it changes and has confidence above 90%.
    if lastLabel is not None and lastLabel != label and score > 0.9:
        speak_label(label)

    # Speak outloud what you are looking at when e is pressed.
    if cv2.waitKey(1) == ord('e'):
        speak_label(label)

    # Window management
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()