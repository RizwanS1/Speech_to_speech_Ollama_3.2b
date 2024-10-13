import pyttsx3
import sounddevice as sd
import vosk
import json
from langchain_ollama import OllamaLLM
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import threading

# Initialize the LLM (Llama3 in this case)
model = OllamaLLM(model="llama3.2")

# Initialize TTS (pyttsx3)
engine = pyttsx3.init()

# Decrease the speed of the speech (adjust the rate)
rate = engine.getProperty('rate')  # Get the default speech rate
engine.setProperty('rate', rate - 50)  # Decrease speed (adjust this value as needed)

# List available voices
voices = engine.getProperty('voices')

# Set female voice (find the correct index for a female voice on your system)
for voice in voices:
    if "female" in voice.name.lower():  # Try to find a female voice
        engine.setProperty('voice', voice.id)
        break

# Initialize Vosk Model for Speech Recognition
vosk_model = vosk.Model("C:\\Users\\shoai\\Downloads\\vosk-model-small-en-us-0.15")

# Global variable for webcam capture
cap = None

def start_webcam():
    """Start the webcam and display the video feed."""
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera

    def show_frame():
        """Capture and display the webcam feed."""
        ret, frame = cap.read()  # Read a frame from the webcam
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = Image.fromarray(frame)  # Create an image from the frame
            img_tk = ImageTk.PhotoImage(image=img)  # Convert to PhotoImage
            video_label.imgtk = img_tk  # Keep a reference to avoid garbage collection
            video_label.configure(image=img_tk)  # Update the label with the new image
        video_label.after(10, show_frame)  # Call this function again after 10 ms

    show_frame()  # Start the frame display

def stop_webcam():
    """Release the webcam."""
    global cap
    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()

# Function to capture speech from the microphone and convert it to text using Vosk
def stt():
    """Function for continuous Speech to Text using the microphone input."""
    rec = vosk.KaldiRecognizer(vosk_model, 16000)

    with sd.RawInputStream(samplerate=16000, blocksize=2000, dtype='int16', channels=1) as stream:
        print("Listening...")
        while True:
            data, _ = stream.read(4000)
            data_bytes = bytes(data)  # Convert to bytes
            
            # Check if any speech has been recognized
            if rec.AcceptWaveform(data_bytes):
                result = rec.Result()
                text = json.loads(result).get('text', '')
                if text:
                    print(f"User: {text}")
                    return text  # Return recognized text once speech is detected

# Function to convert text to speech and directly play it
def tts_speak(response_text):
    """Convert the bot's response to speech and play it directly."""
    engine.say(response_text)  # Directly play the response
    engine.runAndWait()  # Wait for the engine to finish speaking

# Function to handle speech-to-speech interaction
def speech_to_speech():
    print("Say something...")
    while True:
        # 1. Convert speech to text using the microphone (no timeout)
        user_text = stt()

        # Exit condition if user says "exit" or "quit"
        if user_text and user_text.lower() in ["exit", "quit", "stop"]:
            print("Exiting...")
            stop_webcam()  # Stop the webcam feed if exiting
            break

        if user_text:  # Only proceed if there's input
            # 2. Invoke the LLM to generate a response
            response = model.invoke(input=user_text)

            # Since the response is a string, use it directly
            bot_response = response  # LLM output is a string

            print(f"Bot: {bot_response}")

            # 3. Convert the bot's text response to speech and play it directly
            tts_speak(bot_response)

# Function to trigger LLM response when button is pressed
def button_callback():
    # Run the main function in a separate thread to avoid blocking the GUI
    threading.Thread(target=speech_to_speech).start()
    start_webcam()  # Start the webcam feed when starting speech interaction

# Create the GUI window
window = tk.Tk()
window.title("LLM Assistant")

# Add a label to the window
label = tk.Label(window, text="Press the button to start speaking:")
label.pack(pady=10)

# Add a button to the window
start_button = tk.Button(window, text="Start", command=button_callback)
start_button.pack(pady=10)

# Add a label for the webcam feed
video_label = tk.Label(window)
video_label.pack(pady=10)

# Start the Tkinter main loop
window.mainloop()

# Release the webcam when the program exits
stop_webcam()
