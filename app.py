from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import logging
import os

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Allow CORS for all origins (you can restrict it if needed)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Lazy loading: Model loads only when needed
story_generator = None

def load_model():
    global story_generator
    if story_generator is None:  # Load only if not already loaded
        logging.info("Loading GPT-2 model...")
        try:
            story_generator = pipeline("text-generation", model="gpt2")
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
    return story_generator

# Clean the text input or output
def clean_text(text):
    logging.debug(f"Original text: {text}")
    cleaned_text = ''.join([char if char.isalnum() or char.isspace() or char in ['.', ',', '?', '!'] else ' ' for char in text])
    logging.debug(f"Cleaned text: {cleaned_text}")
    return cleaned_text.strip()

# Generate structured story with beginning, middle, and end using title, characters, and story type
def generate_story(title, characters, story_type):
    story_gen = load_model()  # Load model lazily

    if not story_gen:
        return ["Error: Model failed to load."]

    story_parts = {
        "beginning": (
            f"Title: {title}\n"
            f"Characters: {characters}\n"
            f"Story Type: {story_type}\n\n"
            f"Once upon a time, {characters} were living their usual lives until something unexpected happened "
            f"that set them on a new path in this story. "
        ),
        "middle": (
            f"As the story unfolded in this tale, {characters} encountered numerous challenges. "
            "They met new people and discovered important truths. Every step tested their strength, intelligence, "
            "and courage, but they pressed on, determined to overcome every obstacle. "
        ),
        "end": (
            f"In the end, after all the struggles and adventures in this journey, {characters} found what "
            "they were looking for. They learned valuable lessons, grew as individuals, and their journey finally led them "
            "to a hopeful new beginning filled with promise and wisdom."
        )
    }
    
    story_paragraphs = []
    for key, prompt in story_parts.items():
        try:
            # Generate the paragraph for each part
            generated_text = story_gen(prompt, max_length=800, num_return_sequences=1, truncation=True)[0]["generated_text"]
            clean_paragraph = clean_text(generated_text)
            story_paragraphs.append(clean_paragraph)
        except Exception as e:
            logging.error(f"Error generating {key} part: {e}")
            story_paragraphs.append(f"Error generating {key} part.")

    return story_paragraphs

@app.route('/generate_story', methods=['POST'])
def generate_story_endpoint():
    try:
        data = request.json
        title = data.get("title", "Untitled Story")
        characters = data.get("characters", "Unknown Characters")
        story_type = data.get("storyType", "General")

        logging.info(f"Received request: title={title}, characters={characters}, storyType={story_type}")
        
        # Generate story paragraphs based on title, characters, and story type
        paragraphs = generate_story(title, characters, story_type)

        story_pages = [{"type": "text", "content": paragraph} for paragraph in paragraphs]

        return jsonify({"pages": story_pages})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/')
def home():
    return "Story Composer API is running!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Set to 10000 to match Render logs
    app.run(host="0.0.0.0", port=port)
