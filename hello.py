from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer for English to Hindi translation
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text
def translate_to_hindi(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    # Generate the translation
    translated = model.generate(**inputs)
    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Get input from the user
user_input = input("Please enter text to translate to Hindi: ")
translated_output = translate_to_hindi(user_input)

print("Translated text in Hindi:", translated_output)
