import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def convert_speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except FileNotFoundError:
        print(f"File '{audio_file}' not found.")
        raise
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        raise
    except sr.UnknownValueError:
        print("Unknown error occurred during speech-to-text conversion")
        raise

def summarize_text(text):
    try:
        # Load the summarization model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn") #force_download=True )
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")#force_download=True)

        # Encode the input text and generate the summary
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error in text summarization: {e}")
        raise

def process_audio_file(audio_file):
    print("Converting speech to text...")
    try:
        meeting_text = convert_speech_to_text(audio_file)
        print("Transcribed Text:\n", meeting_text)
    except Exception as e:
        print(f"Error in speech-to-text conversion: {e}")
        return
    
    print("Summarizing text...")
    try:
        summary = summarize_text(meeting_text)
        print("Meeting Summary:\n", summary)
    except Exception as e:
        print(f"Error in text summarization: {e}")

def process_text_file(text_file):
    if not os.path.isfile(text_file):
        print(f"File '{text_file}' not found.")
        return
    
    print("Reading text file...")
    try:
        with open(text_file, 'r') as file:
            meeting_text = file.read()
        print("Text from file:\n", meeting_text)
    except Exception as e:
        print(f"Error reading text file: {e}")
        return
    
    print("Summarizing text...")
    try:
        summary = summarize_text(meeting_text)
        print("Meeting Summary:\n", summary)
    except Exception as e:
        print(f"Error in text summarization: {e}")

def process_file(file_to_process):
    if not os.path.isfile(file_to_process):
        print(f"File '{file_to_process}' not found.")
        return

    # Check the file extension to determine if it's an audio or text file
    if file_to_process.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        process_audio_file(file_to_process)
    elif file_to_process.lower().endswith('.txt'):
        process_text_file(file_to_process)
    else:
        print("Unsupported file type. Please provide an audio file (.wav, .mp3, .flac, .m4a) or a text file (.txt).")

def main():
    # Specify the file you want to process
    file_to_process = input("Enter the file path: ")

    process_file(file_to_process)

if __name__ == "__main__":
    main()

