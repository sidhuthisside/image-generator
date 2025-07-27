from flask import Flask, request, jsonify
import os
import base64
from PIL import Image
from io import BytesIO

import google.generativeai as genai
from google.cloud import aiplatform
from google.oauth2 import service_account

app = Flask(__name__)

# ===== ENV VARS =====
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION")
CREDENTIALS_PATH = "key.json"

# ===== INIT GOOGLE GENAI ONCE =====
genai.configure(api_key=GEMINI_API_KEY)

# ===== FUNCTIONS =====

def initialize_gemini():
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        print(f"Gemini init error: {e}")
        return None

def enhance_prompt(gemini_model, original_prompt):
    try:
        chat = gemini_model.start_chat()
        response = chat.send_message(
            f"Original prompt: {original_prompt}\n\nPlease enhance this for image generation.",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                candidate_count=1,
                max_output_tokens=200,
            )
        )
        return response.text.strip().strip('"')
    except Exception as e:
        print(f"Prompt enhancement error: {e}")
        return f"{original_prompt}, ultra HD, 8K resolution, professional photography"

def generate_image(prompt, use_enhanced=False):
    gemini_model = initialize_gemini()
    final_prompt = enhance_prompt(gemini_model, prompt) if use_enhanced else prompt

    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

    client = aiplatform.gapic.PredictionServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )

    endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/imagegeneration"
    response = client.predict(
        endpoint=endpoint,
        instances=[{
            "prompt": final_prompt,
            "sampleCount": 1,
            "seed": 42,
            "guidanceScale": 7.5
        }]
    )

    if response.predictions:
        image_data = base64.b64decode(response.predictions[0]['bytesBase64Encoded'])
        return base64.b64encode(image_data).decode("utf-8")
    else:
        return None

# ===== ROUTES =====

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt")
    use_enhanced = data.get("enhance", False)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    image_base64 = generate_image(prompt, use_enhanced)
    if image_base64:
        return jsonify({"image": image_base64})
    else:
        return jsonify({"error": "Generation failed"}), 500

@app.route('/')
def index():
    return "✅ Flask API is running"

# ===== RUN =====
if __name__ == '__main__':
    app.run(debug=True)
