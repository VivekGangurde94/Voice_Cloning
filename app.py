import os
import torch
import torchaudio
from flask import Flask, request, jsonify, send_file
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from audio_conversion import convert_to_wav
import uuid

app = Flask(__name__)

# Load the model once and reuse
print("Loading model...")
config = XttsConfig()
config.load_json("D:/xtts2-ui-main/tts_model/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="D:/xtts2-ui-main/tts_model/")
model.cuda()
print("Model loaded.")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data['text']
    language = data['language']
    speaker = data['speaker']
    temperature = data.get('temperature', 0.7)

    try:
        # Ensure the speaker file is in WAV format
        speaker_wav = f"targets/{uuid.uuid4().hex}.wav"
        convert_to_wav(f"targets/{speaker}", speaker_wav)

        # Compute speaker latents
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
        
        out = model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
        )

        # Save the output audio in WAV format
        output_file = f"output_{uuid.uuid4().hex}.wav"
        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        # Remove the temporary speaker wav file
        os.remove(speaker_wav)

        return send_file(output_file, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
