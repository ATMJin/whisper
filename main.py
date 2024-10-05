import whisper
from whisper_transcribe import WhisperTranscriber

# 選擇 model: https://github.com/openai/whisper#available-models-and-languages
model = whisper.load_model("large-v3").to("cuda")  # large: 2.88G

audio_dir = "F:/Voice//01_WAV版"
audio_name = "!.wav"

trans = WhisperTranscriber(model)
trans.transcribe(audio_dir, audio_name)
