from datetime import datetime
from flask import Flask, request, jsonify
import os
import speech_recognition as sr
from googletrans import Translator
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load model and configuration
config = XttsConfig()
config.load_json("model/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="model")

@app.route('/process_clip', methods=['POST'])
def process_clip():
    data = {}
    try:
        audio_clip = request.files['audio_clip']
        target_language = request.form['target_language']
        source_language = request.form['source_language']

        # Define paths for saving the clips
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_clip_folder = "input_clips/"
        output_clip_folder = "output_clips/"
        input_clip_filename = f"Lang_{source_language}_{current_time}_input.wav"
        output_clip_filename = f"Lang_{target_language}_{current_time}_output.wav"
        input_clip_path = os.path.join(input_clip_folder, input_clip_filename)
        output_clip_path = os.path.join(output_clip_folder, output_clip_filename)

        # Ensure directories exist
        os.makedirs(input_clip_folder, exist_ok=True)
        os.makedirs(output_clip_folder, exist_ok=True)

        # Save the audio file to the input folder
        with open(input_clip_path, 'wb') as f:
            f.write(audio_clip.read())

        # Perform speech recognition with the provided source language
        recognized_text = recognize_speech(input_clip_path, source_language)

        # Translate recognized text
        translated_text = translate_text(recognized_text, target_language)

        # Synthesize audio and save to the output folder
        output_file = synthesize_audio(translated_text, input_clip_path, source_language, target_language, output_clip_path)

        # Read the synthesized audio file and encode it in base64
        with open(output_file, 'rb') as audio_file:
            data["audio_clip"] = base64.b64encode(audio_file.read()).decode('utf-8')

        data["translated_text"] = translated_text
        data["recognized_text"] = recognized_text

    except Exception as e:
        data["error"] = str(e)
        return jsonify(data), 500

    return jsonify(data)

def recognize_speech(audio_clip, language):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_clip) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='ru-RU')
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

def translate_text(text, target_language):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def synthesize_audio(text, audio_file, source_language, target_language, output_clip_path):
    # Compute speaker latents
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_file)

    # Inference
    chunks = model.inference_stream(text, source_language, gpt_cond_latent, speaker_embedding)

    wav_chunks = []
    for chunk in chunks:
        wav_chunks.append(chunk)

    wav = torch.cat(wav_chunks, dim=0)

    # Save the audio file to the output folder
    torchaudio.save(output_clip_path, wav.squeeze().unsqueeze(0).cpu(), 24000)
    print("Audio saved")

    return output_clip_path

if __name__ == '__main__':
    app.run(debug=True)
