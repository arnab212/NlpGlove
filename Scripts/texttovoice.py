# Voice-to-Grasp - utility: generates short spoken test clips with text-to-speech (pyttsx3), e.g. for
# timing the speech-recognition stage without a participant (Revision_ISR_2026/05_latency_benchmark.py).

import pyttsx3

def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("Available voices:")
    for index, voice in enumerate(voices):
        print(f"{index}: {voice.name} ({voice.id})")

def text_to_voice_save(text, filename, voice_index=0):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # Set selected voice by index
    if 0 <= voice_index < len(voices):
        engine.setProperty('voice', voices[voice_index].id)
    else:
        print("Invalid voice index. Using default voice.")

    engine.save_to_file(text, filename)
    engine.runAndWait()
    print(f"Saved voice audio to {filename}")

if __name__ == "__main__":
    list_voices()

    text = "Releasing now"
    filename = "releasing.mp3"
    # chosen_voice_index = int(input("Enter the voice index to use: "))
    text_to_voice_save(text, filename, 2)
2