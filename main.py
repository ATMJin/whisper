import whisper
from whisper_transcribe import WhisperTranscribe

# 選擇 model: https://github.com/openai/whisper#available-models-and-languages
model = whisper.load_model("large-v3").to("cuda")  # large: 2.88G
