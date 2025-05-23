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
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer, AutoModelForTextToWaveform, PreTrainedTokenizer, PreTrainedModel
except ImportError:
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.error("Required libraries not found. Please ensure transformers, torch, etc. are installed.")
    logging.error("Install instructions: pip install -r requirements.txt")
    exit(1)

# --- Configuration ---
load_dotenv()
PARLER_MODEL_ID = "parler-tts/parler-tts-mini-v1.1"
HOST = os.getenv("FLASK_HOST", "127.0.0.1")
PORT = int(os.getenv("FLASK_PORT_PARLER", "3004"))
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "1000"))
DEFAULT_DESCRIPTION = os.getenv(
    "DEFAULT_DESCRIPTION",
    "A female speaker delivers a slightly expressive and animated speech. The recording is of very high quality."
)

# --- Flask App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s-parler - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s-parler - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not app.logger.handlers: # Avoid adding duplicate handlers
    app.logger.addHandler(handler)
app.logger.propagate = False # Prevent duplicate logs in root logger

# --- Global Variables for Model ---
device: Optional[torch.device] = None
model: Optional[PreTrainedModel] = None
tokenizer: Optional[PreTrainedTokenizer] = None

def load_model() -> None:
    """
    Loads the specified Hugging Face TTS model and tokenizer into global variables.
    Determines the appropriate device (CUDA, MPS, CPU) and handles potential errors.
    Forces float32 dtype for stability, especially on MPS.
    """
    global device, model, tokenizer
    app.logger.info(f"Attempting to load Parler-TTS model: {PARLER_MODEL_ID}")

    try:
        forced_device = os.getenv("TORCH_DEVICE")
        if forced_device:
            device = torch.device(forced_device)
            app.logger.info(f"Using forced device from TORCH_DEVICE: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
             # Force CPU even if MPS is available due to potential compatibility issues
             # like "Output channels > 65536 not supported at the MPS device."
             device = torch.device("cpu")
             app.logger.warning("MPS device detected, but forcing CPU due to potential compatibility issues.")
             # device = torch.device("mps")
             # app.logger.info("MPS device detected.")
        else:
            device = torch.device("cpu")

        dtype = torch.float32
        app.logger.warning(f"Forcing data type: {dtype} for model loading (prioritizing stability).")

        app.logger.info(f"Loading Parler-TTS model '{PARLER_MODEL_ID}' with dtype {dtype} onto device '{device}'...")

        tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL_ID)
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            PARLER_MODEL_ID,
            torch_dtype=dtype
        ).to(device)

        model.eval()
        app.logger.info("Parler-TTS model and tokenizer loaded successfully.")

    except Exception as e:
        app.logger.error(f"Fatal error loading Parler-TTS model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load Parler-TTS model '{PARLER_MODEL_ID}': {e}")


# --- Synthesis API Endpoint ---
@app.route('/synthesize', methods=['POST'])
def synthesize() -> Union[Tuple[Response, int], Response]:
    """
    API endpoint to synthesize speech from text.
    Expects a JSON payload with 'text' and optionally 'description' (required for Parler-TTS).
    Returns a WAV audio file or a JSON error message.
    """
    if model is None or tokenizer is None:
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
    description: Optional[str] = data.get('description')
    is_parler = isinstance(model, ParlerTTSForConditionalGeneration)
    if is_parler and not description:
        description = DEFAULT_DESCRIPTION
        app.logger.info(f"No description provided for Parler-TTS model, using default: '{description[:50]}...'")
    elif description:
         app.logger.info(f"Using provided description: '{description[:50]}...'")


    if not text_to_speak or not isinstance(text_to_speak, str):
        app.logger.warning("Missing or invalid 'text' parameter in request.")
        return jsonify({"error": "Missing or invalid 'text' parameter in JSON body"}), 400

    if len(text_to_speak) > MAX_TEXT_LENGTH:
        app.logger.warning(f"Input text length ({len(text_to_speak)}) exceeds limit ({MAX_TEXT_LENGTH}).")
        return jsonify({"error": f"Input text exceeds maximum length of {MAX_TEXT_LENGTH} characters"}), 413 # Payload Too Large

    log_desc = f" Description: '{description[:50]}...'" if description else ""
    app.logger.info(f"Received synthesis request. Text length: {len(text_to_speak)}.{log_desc}")
    app.logger.debug(f"Full text: '{text_to_speak}'")

    try:
        assert tokenizer is not None and model is not None and device is not None

        app.logger.info("Starting audio generation...")
        with torch.no_grad():
            if is_parler:
                assert description is not None # Should have default if not provided
                app.logger.debug("Tokenizing description and prompt for Parler-TTS...")
                description_tokens = tokenizer(description, return_tensors="pt").to(device)
                prompt_tokens = tokenizer(text_to_speak, return_tensors="pt").to(device)

                app.logger.debug(f"Parler Description input_ids shape: {description_tokens.input_ids.shape}")
                app.logger.debug(f"Parler Prompt input_ids shape: {prompt_tokens.input_ids.shape}")

                output = model.generate(
                    input_ids=description_tokens.input_ids,
                    prompt_input_ids=prompt_tokens.input_ids,
                    attention_mask=description_tokens.attention_mask
                ).to(device)
                waveform = output
           
        app.logger.info("Audio generation finished.")

        app.logger.debug(f"Moving generated waveform tensor (shape: {waveform.shape}, dtype: {waveform.dtype}) to CPU...")
        audio_arr = waveform.cpu().numpy().squeeze().astype("float32")
        app.logger.debug(f"Tensor moved to CPU and converted to NumPy array (shape: {audio_arr.shape}, dtype: {audio_arr.dtype}).")

        min_val, max_val, mean_val = audio_arr.min(), audio_arr.max(), audio_arr.mean()
        app.logger.info(f"Audio array stats before saving - Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        if abs(min_val) > 1.0 or abs(max_val) > 1.0:
            app.logger.warning("Audio array values fall outside the typical [-1.0, 1.0] range. Clipping might occur during PCM conversion.")

        try:
            sampling_rate = model.config.sampling_rate
            app.logger.info(f"Using sampling rate from model config: {sampling_rate} Hz")
        except AttributeError:
            app.logger.warning("Could not automatically determine sampling rate from model config. Falling back to a default (e.g., 16000 Hz). This might be incorrect.")
            # Fallback or require configuration if needed. For now, let's assume 16kHz is a common default.
            sampling_rate = 16000

        app.logger.debug("Writing audio array to WAV buffer...")
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, sampling_rate, format='WAV', subtype='PCM_16') # Common WAV format
        buffer.seek(0) # Rewind buffer to the beginning for reading
        app.logger.debug("WAV buffer created successfully.")

        app.logger.info("Speech synthesis successful. Sending audio response.")

        return send_file(
            buffer,
            mimetype='audio/wav',
            as_attachment=False # Send as inline content, not download
            # Consider adding attachment headers if download is preferred:
            # as_attachment=True,
            # download_name='synthesis.wav'
        )

    except Exception as e:
        app.logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during speech synthesis"}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        load_model()

        app.logger.info("--- Parler-TTS Server Configuration ---")
        app.logger.info(f"Model ID: {PARLER_MODEL_ID}")
        app.logger.info(f"Device: {device}")
        app.logger.info(f"Model Data Type: {model.dtype if model else 'N/A'}")
        app.logger.info(f"Max Text Length: {MAX_TEXT_LENGTH}")
        if isinstance(model, ParlerTTSForConditionalGeneration):
            app.logger.info(f"Default Description (for Parler): {DEFAULT_DESCRIPTION}")
        app.logger.info(f"Running Flask server on http://{HOST}:{PORT}")
        app.logger.info("-----------------------------")

        # Start the Flask development server
        # Set debug=False for production/stable environments or when using a production WSGI server
        # Set debug=True for development to get auto-reloading and detailed error pages
        # Use threaded=False if you suspect thread-safety issues, though Flask+Torch usually handles this
        app.run(host=HOST, port=PORT, debug=False, threaded=True)

    except RuntimeError as e:
        app.logger.critical(f"Application startup failed during model loading: {e}", exc_info=False)
        exit(1)
    except Exception as e:
        app.logger.critical(f"An unexpected error occurred during application startup: {e}", exc_info=True)
        exit(1)
