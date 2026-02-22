import pyttsx3

engine = pyttsx3.init()

def speak_label(label: str):
    """
    Converts a label, i.e. 'This is a photo of banana.'
    to a sentence, i.e. 'You are looking at a banana.'
    Uses TTS to speak it out loud.
    """
    prefix = "This is a photo of "
    if label.startswith(prefix):
        label = label[len(prefix):]
    spoken = f"You are looking at a {label}"

    engine.say(spoken)
    engine.runAndWait()