import os
import logging
import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

from flask import Flask, request, send_file, jsonify, Response
from dotenv import load_dotenv
import torch
import soundfile as sf

# --- Dependencies Check and Import ---
try:
    from dia.model import Dia
except ImportError:
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error("Required libraries not found. Please ensure dia-tts, torch, soundfile etc. are installed.")
    logging.error("Check installation instructions for Dia-TTS and requirements.txt")
    exit(1)

# --- Configuration ---
load_dotenv()
DIA_MODEL_ID = "nari-labs/Dia-1.6B"
DIA_SAMPLING_RATE = 44100
HOST = os.getenv("FLASK_HOST", "127.0.0.1")
PORT = int(os.getenv("FLASK_PORT_DIA", "3005"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "1000"))

# --- Flask App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s-dia - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s-dia - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not app.logger.handlers: # Avoid adding duplicate handlers
    app.logger.addHandler(handler)
app.logger.propagate = False # Prevent duplicate logs in root logger

# --- Global Variables for Model ---
device: Optional[torch.device] = None
model: Optional[Dia] = None

def load_model() -> None:
    """
    Loads the Dia-TTS model into global variables.
    Determines the appropriate device (CUDA, MPS, CPU) and handles potential errors.
    Forces float32 dtype for stability.
    """
    global device, model
    app.logger.info(f"Attempting to load Dia-TTS model: {DIA_MODEL_ID}")

    try:
        forced_device = os.getenv("TORCH_DEVICE")
        if forced_device:
            device = torch.device(forced_device)
            app.logger.info(f"Using forced device from TORCH_DEVICE: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
             # Force CPU even if MPS is available due to potential compatibility issues
             device = torch.device("cpu")
             app.logger.warning("MPS device detected, but forcing CPU due to potential compatibility issues.")
        else:
            device = torch.device("cpu")

        dtype = torch.float32
        app.logger.warning(f"Using data type: {dtype} for model loading (check Dia documentation if issues arise).")

        app.logger.info(f"Loading Dia-TTS model '{DIA_MODEL_ID}' with dtype {dtype} onto device '{device}'...")

        model = Dia.from_pretrained(DIA_MODEL_ID)
        if hasattr(model, 'eval'):
            model.eval()

        app.logger.info("Dia-TTS model loaded successfully.")

    except Exception as e:
        app.logger.error(f"Fatal error loading Dia-TTS model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Dia-TTS model '{DIA_MODEL_ID}': {e}")


# --- Synthesis API Endpoint ---
@app.route('/synthesize', methods=['POST'])
def synthesize() -> Union[Tuple[Response, int], Response]:
    """
    API endpoint to synthesize speech from text using Dia-TTS.
    Expects a JSON payload with 'text'.
    Returns a WAV audio file or a JSON error message.
    """
    if model is None:
         app.logger.error("Synthesis request received, but model is not loaded.")
         return jsonify({"error": "Model not ready. Please try again later or check server logs."}), 503 # Service Unavailable

    if not request.is_json:
        app.logger.warning("Received non-JSON request.")
        return jsonify({"error": "Invalid request format: Content-Type must be application/json"}), 400 # Bad Request

    data: Optional[Dict[str, Any]] = request.get_json()
    if not data:
         app.logger.warning("Received empty JSON payload.")
         return jsonify({"error": "Empty JSON payload received"}), 400

    text_to_speak: Optional[str] = data.get('text')

    if not text_to_speak or not isinstance(text_to_speak, str):
        app.logger.warning("Missing or invalid 'text' parameter in request.")
        return jsonify({"error": "Missing or invalid 'text' parameter in JSON body"}), 400

    if len(text_to_speak) > MAX_TEXT_LENGTH:
        app.logger.warning(f"Input text length ({len(text_to_speak)}) exceeds limit ({MAX_TEXT_LENGTH}).")
        return jsonify({"error": f"Input text exceeds maximum length of {MAX_TEXT_LENGTH} characters"}), 413 # Payload Too Large

    app.logger.info(f"Received synthesis request. Text length: {len(text_to_speak)}.")
    app.logger.debug(f"Full text: '{text_to_speak}'")

    try:
        assert model is not None and device is not None

        app.logger.info("Starting audio generation with Dia-TTS...")
        with torch.no_grad(): # Ensure gradients are not computed
            app.logger.debug("Generating audio with Dia model...")
            output_tensor = model.generate(text_to_speak)

            if not isinstance(output_tensor, torch.Tensor):
                 app.logger.error(f"Dia model output is not a tensor, type: {type(output_tensor)}")
                 raise TypeError("Dia model did not return a tensor.")

            waveform = output_tensor.to('cpu')

        app.logger.info("Audio generation finished.")

        app.logger.debug(f"Moving generated waveform tensor (shape: {waveform.shape}, dtype: {waveform.dtype}) to CPU...")
        audio_arr = waveform.numpy().squeeze().astype("float32")
        app.logger.debug(f"Tensor moved to CPU and converted to NumPy array (shape: {audio_arr.shape}, dtype: {audio_arr.dtype}).")

        min_val, max_val, mean_val = audio_arr.min(), audio_arr.max(), audio_arr.mean()
        app.logger.info(f"Audio array stats before saving - Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        if abs(min_val) > 1.0 or abs(max_val) > 1.0:
            app.logger.warning("Audio array values fall outside the typical [-1.0, 1.0] range. Clipping might occur during PCM conversion.")

        sampling_rate = DIA_SAMPLING_RATE
        app.logger.info(f"Using sampling rate for Dia: {sampling_rate} Hz")

        app.logger.debug("Writing audio array to WAV buffer...")
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, sampling_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0) # Rewind buffer to the beginning for reading
        app.logger.debug("WAV buffer created successfully.")

        app.logger.info("Speech synthesis successful. Sending audio response.")

        return send_file(
            buffer,
            mimetype='audio/wav',
            as_attachment=False # Send as inline content
        )

    except Exception as e:
        app.logger.error(f"Error during Dia-TTS synthesis: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during speech synthesis"}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        load_model()
        app.logger.info("--- Dia-TTS Server Configuration ---")
        app.logger.info(f"Model ID: {DIA_MODEL_ID}")
        app.logger.info(f"Device: {device}") # Device is set in load_model
        app.logger.info(f"Model Data Type: {model.dtype if hasattr(model, 'dtype') else 'N/A'}")
        app.logger.info(f"Sampling Rate: {DIA_SAMPLING_RATE}")
        app.logger.info(f"Max Text Length: {MAX_TEXT_LENGTH}")
        app.logger.info(f"Running Flask server on http://{HOST}:{PORT}")
        app.logger.info("-----------------------------")

        app.run(host=HOST, port=PORT, debug=False, threaded=True)

    except RuntimeError as e:
        app.logger.critical(f"Application startup failed during model loading: {e}", exc_info=False)
        exit(1)
    except Exception as e:
        app.logger.critical(f"An unexpected error occurred during application startup: {e}", exc_info=True)
        exit(1)
