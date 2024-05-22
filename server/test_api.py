import requests
import base64

# Set the URL of your Flask API endpoint
url = 'http://127.0.0.1:5000/process_clip'

# Set the path to the audio file you want to send
audio_file_path = 'input_clips\Lang_ru_20240518_035218_input.wav'
files = {}

# Open the audio file
with open(audio_file_path, 'rb') as audio_file:
    files = {'audio_clip': audio_file}

    # Add form data including target language and source language
    form_data = {'target_language': 'es', 'source_language': 'en'}

    # Send the POST request with both files and form data
    response = requests.post(url, files=files, data=form_data)

# Print the recognized text returned by the server
print("Response:", response.json())
