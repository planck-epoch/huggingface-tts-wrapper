# Hugging Face TTS API Wrapper

A simple Flask-based API wrapper to serve Text-to-Speech (TTS) models from the Hugging Face Hub.

## Features

*   Provides a `/synthesize` endpoint to generate speech from text.
*   Configurable TTS model via environment variables.
*   Supports different compute devices (CUDA, MPS, CPU).
*   Handles basic error checking and logging.

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
    *Note: `parler-tts` is installed directly from GitHub via the requirements file.*

4.  **Configure the model (Optional):**
    Create a `.env` file in the project root directory by copying the example:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file to specify the Hugging Face model ID you want to use.

## Configuration

The application uses environment variables for configuration. You can set these in a `.env` file or directly in your environment.

*   `MODEL_ID`: The Hugging Face model identifier for the TTS model (e.g., `parler-tts/parler-tts-mini-v1.1`, `nari-labs/dia`). Defaults to `parler-tts/parler-tts-mini-v1.1`.
*   `FLASK_HOST`: The host address for the Flask server. Defaults to `127.0.0.1`.
*   `FLASK_PORT`: The port for the Flask server. Defaults to `3003`.
*   `MAX_TEXT_LENGTH`: Maximum number of characters allowed in the input text. Defaults to `1000`.
# Removed DEFAULT_DESCRIPTION

## API Usage

### Endpoint: `/synthesize`

*   **Method:** `POST`
*   **Content-Type:** `application/json`
*   **Request Body:**
    ```json
    {
      "text": "The text you want to convert to speech."
    }
    ```
    *Note: Some models might support additional parameters like speaker embeddings. Check the specific model's documentation on Hugging Face.*
*   **Success Response:**
    *   **Code:** `200 OK`
    *   **Content-Type:** `audio/wav`
    *   **Body:** The raw WAV audio data.
*   **Error Responses:**
    *   `400 Bad Request`: Invalid JSON format, missing 'text', or invalid 'text' type.
    *   `413 Payload Too Large`: Input 'text' exceeds `MAX_TEXT_LENGTH`.
    *   `500 Internal Server Error`: Error during synthesis process.
    *   `503 Service Unavailable`: Model is not loaded or ready.

### Example Request (using `curl`):

```bash
curl -X POST http://127.0.0.1:3003/synthesize \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world, this is a test."}' \
     --output output.wav
```
This command sends a request to the server and saves the resulting audio to `output.wav`.

## Running the Server

```bash
python app.py
```
The server will start, load the configured model, and listen for requests on the specified host and port.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
