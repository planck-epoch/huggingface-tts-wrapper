# Hugging Face TTS API Wrappers

A project providing simple Flask-based API wrappers to serve specific Text-to-Speech (TTS) models from the Hugging Face ecosystem. This project now contains two separate applications: one for Parler-TTS and one for Dia-TTS.

## Features

*   **`app-parler-tts.py`**: Serves the `parler-tts/parler-tts-mini-v1.1` model.
    *   Provides a `/synthesize` endpoint requiring `text` and `description`.
    *   Runs on port 3004 by default (configurable via `FLASK_PORT_PARLER`).
*   **`app-dia-tts.py`**: Serves the `nari-labs/Dia-1.6B` model.
    *   Provides a `/synthesize` endpoint requiring `text`.
    *   Runs on port 3005 by default (configurable via `FLASK_PORT_DIA`).
*   Configurable host, ports, and text limits via environment variables.
*   Supports different compute devices (CUDA, CPU, with MPS forced to CPU for compatibility).
*   Handles basic error checking and logging for each application.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd huggingface-tts-wrapper
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Make sure you have PyTorch installed according to your system's requirements (CPU/CUDA). See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
    Then, install the project requirements:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `parler-tts` and `dia-tts` dependencies are installed directly from GitHub via the requirements file.*

4.  **Configure Environment Variables (Optional):**
    Create a `.env` file in the project root directory by copying the example:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file to customize ports, host, etc. (see Configuration section below). The specific model for each app is now hardcoded.

## Configuration

The applications use environment variables for configuration. You can set these in a `.env` file or directly in your environment.

*   `FLASK_HOST`: The host address for the Flask servers. Defaults to `127.0.0.1`.
*   `FLASK_PORT_PARLER`: The port for the Parler-TTS Flask server. Defaults to `3004`.
*   `FLASK_PORT_DIA`: The port for the Dia-TTS Flask server. Defaults to `3005`.
*   `MAX_TEXT_LENGTH`: Maximum number of characters allowed in the input text for both apps. Defaults to `1000`.
*   `DEFAULT_DESCRIPTION`: The default speaker description/prompt to use for Parler-TTS if none is provided in the request.
*   `TORCH_DEVICE`: (Optional) Force a specific device (e.g., "cuda", "cpu"). If commented out or empty, the script will attempt auto-detection (CUDA > CPU). MPS is detected but CPU is forced due to compatibility issues.

## API Usage

Each application runs on a different port.

### Parler-TTS (`app-parler-tts.py`) Endpoint: `/synthesize`

*   **Method:** `POST`
*   **Content-Type:** `application/json`
*   **Request Body:**
    ```json
    {
      "text": "The text you want to convert to speech.",
      "description": "A description of the desired voice characteristics (e.g., 'A female speaker with a clear voice.'). If omitted, the `DEFAULT_DESCRIPTION` from the configuration is used."
    }
    ```
*   **Success Response:**
    *   **Code:** `200 OK`
    *   **Content-Type:** `audio/wav`
    *   **Body:** The raw WAV audio data.

### Dia-TTS (`app-dia-tts.py`) Endpoint: `/synthesize`

*   **Method:** `POST`
*   **Content-Type:** `application/json`
*   **Request Body:**
    ```json
    {
      "text": "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
    }
    ```
    *Note: Dia-TTS expects specific formatting for dialogue turns (e.g., `[S1]`, `[S2]`).*
*   **Success Response:**
    *   **Code:** `200 OK`
    *   **Content-Type:** `audio/wav`
    *   **Body:** The raw WAV audio data (at 44100 Hz).

### Error Responses (Both Apps):

*   `400 Bad Request`: Invalid JSON format, missing required parameters, or invalid parameter types.
*   `413 Payload Too Large`: Input 'text' exceeds `MAX_TEXT_LENGTH`.
*   `500 Internal Server Error`: Error during synthesis process.
*   `503 Service Unavailable`: Model is not loaded or ready.

### Example Requests (using `curl`):

```bash
# Example for Parler-TTS (runs on port 3004 by default)
curl -X POST http://127.0.0.1:3004/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world, this is a Parler-TTS test.", "description": "A calm male voice."}' \
     --output output_parler.wav

# Example for Dia-TTS (runs on port 3005 by default)
curl -X POST http://127.0.0.1:3005/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "[S1] This is speaker one testing Dia TTS. [S2] And this is speaker two responding."}' \
     --output output_dia.wav
```
These commands send requests to the respective servers and save the resulting audio.

## Running the Servers

You need to run each application in a separate terminal process.

**Terminal 1 (Parler-TTS):**
```bash
python app-parler-tts.py
```
The server will start, load the Parler-TTS model, and listen on the port specified by `FLASK_PORT_PARLER` (default 3004).

**Terminal 2 (Dia-TTS):**
```bash
python app-dia-tts.py
```
The server will start, load the Dia-TTS model, and listen on the port specified by `FLASK_PORT_DIA` (default 3005).

