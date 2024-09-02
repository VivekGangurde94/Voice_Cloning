import os
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_conversion import convert_to_wav
import warnings 
import uuid
import logging
import re
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model once and reuse
try:
    logger.info("Loading model...")
    config = XttsConfig()
    config.load_json(os.getenv('MODEL_CONFIG_PATH', "D:/tts_model/config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=os.getenv('CHECKPOINT_DIR', "D:/tts_model/"))
    model.cuda()
    logger.info("Model loaded.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise e

def split_text(text, max_length=1000):
    # Split the text into sentences based on punctuation followed by a space
    sentences = re.split(r'(?<=[.!?।]) +', text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if len(sentence) > max_length:
            # Split the sentence into smaller chunks if it exceeds max_length
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_length:
                    if current_chunk:
                        current_chunk += ' ' + word
                    else:
                        current_chunk = word
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = word
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ''
        else:
            # Add the sentence to the current chunk if it fits
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                if current_chunk:
                    current_chunk += ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                # If the current chunk is full, add it to chunks and start a new one
                chunks.append(current_chunk.strip())
                current_chunk = sentence

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def concatenate_audio_clips(clips, output_dir):
    combined_clip = AudioSegment.empty()
    for clip in clips:
        combined_clip += AudioSegment.from_wav(clip)
    output_file = os.path.join(output_dir, f"output_{uuid.uuid4().hex}.wav")
    combined_clip.export(output_file, format="wav")
    return output_file

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json

    # Input validation
    if not all(k in data for k in ("text", "language", "speaker")):
        return jsonify({"error": "Missing required fields in request."}), 400

    text = data['text']
    language = data['language']
    speaker = data['speaker']
    temperature = data.get('temperature', 0.7)
    speed = data.get('speed', 1)

    try:
        # Ensure the speaker file is in WAV format
        speaker_file_path = os.path.join("targets", speaker)
        speaker_wav = f"targets/{uuid.uuid4().hex}.wav"
        if not speaker_file_path.lower().endswith('.wav'):
            convert_to_wav(speaker_file_path, speaker_wav)
        else:
            speaker_wav = speaker_file_path

        # Compute speaker latents
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
        
        output_dir = "outputs"
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Split text into manageable chunks
        chunks = split_text(text, max_length=150)
        audio_clips = []

        for chunk in chunks:
            out = model.inference(
                chunk,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=temperature,
                speed = speed,
            )
            chunk_output_file = os.path.join(output_dir, f"chunk_{uuid.uuid4().hex}.wav")
            torchaudio.save(chunk_output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)
            audio_clips.append(chunk_output_file)

        # Concatenate audio clips
        result_audio_file = concatenate_audio_clips(audio_clips, output_dir)

        # Clean up temporary files
        for clip in audio_clips:
            os.remove(clip)
        if speaker_file_path != speaker_wav:
            os.remove(speaker_wav)

        return send_file(result_audio_file, as_attachment=True)

    except Exception as e:
        logger.error(f"Error during synthesis: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', False))
