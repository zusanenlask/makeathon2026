import cv2
from transformers import pipeline
from PIL import Image
from TTS import speak
from label_manager import LabelManager

# Managers, stream, and model
lm = LabelManager()
checkpoint = "openai/clip-vit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
cap = cv2.VideoCapture("http://10.49.243.172/stream") #"http://10.49.243.172/stream"

# Classifciation labels
coarse_labels = ["pencil", "notebook", "calculator", "motor", "circuit board"]
fine_labels = {
    "motor": ["dc motor", "stepper motor"],
    "circuit board": ["digital temp sensor circuit"]
}
coarse_prompts = lm.add_CLIP_prefix(coarse_labels)
fine_prompts = {k: lm.add_CLIP_prefix(v) for k, v in fine_labels.items()}

# FPS control (run model every SKIP frames)
frame_count = 0
SKIP = 6
disp_coarse_label = ""
disp_coarse_score = 0.0
disp_fine_label = None
disp_fine_score = None
last_spoken_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera feed. Exiting...")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    frame_count += 1
    if frame_count % SKIP == 0:
        # Convert frame to model-readable color format
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Coarse classification
        predictions = detector(pil_image, coarse_prompts)
        coarse_best = predictions[0]
        raw_coarse = lm.strip_CLIP_prefix(coarse_best["label"])
        coarse_score = coarse_best["score"]

        disp_coarse_label = raw_coarse
        disp_coarse_score = coarse_score

        # Fine classification
        disp_fine_label = None
        disp_fine_score = None

        if raw_coarse in fine_prompts:
            fine_predictions = detector(pil_image, fine_prompts[raw_coarse])
            fine_best = fine_predictions[0]
            disp_fine_label = lm.strip_CLIP_prefix(fine_best["label"])
            disp_fine_score = fine_best["score"]

        # Speak if confident (85% coarse, 65% fine)
        final_label = disp_fine_label if disp_fine_label else disp_coarse_label
        if final_label != last_spoken_label and disp_coarse_score > 0.85:
            if disp_fine_score is None or disp_fine_score > 0.65:
                speak(final_label)
                last_spoken_label = final_label

    # Display object type
    frame = cv2.putText(
        frame,
        f"{disp_coarse_label}: {disp_coarse_score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    if disp_fine_label is not None:
        frame = cv2.putText(
            frame,
            f"{disp_fine_label}: {disp_fine_score:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break

    cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()