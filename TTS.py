import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")

def speak(text: str):
    """
    Uses Windows SAPI5 to speak text outloud.
    """
    speaker.Speak(text)