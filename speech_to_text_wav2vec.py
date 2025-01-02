import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

def transcribe_audio_with_wav2vec(audio_file):
    # Load Wav2Vec2 model and processor from Hugging Face
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Load audio
    audio_input, _ = librosa.load(audio_file, sr=16000)

    # Preprocess audio
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)

    # Make prediction
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    # Decode prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    print("Transcription: ", transcription)

# Example usage
if __name__ == "__main__":
    audio_file = "your_audio_file.wav"  # Path to your audio file
    transcribe_audio_with_wav2vec(audio_file)
