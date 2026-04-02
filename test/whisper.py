from faster_whisper import WhisperModel
import os
from openai import OpenAI

if __name__ == "__main__":
    
    base_url = "http://172.18.31.238:8000/v1"
    openai_api_key = "EMPTY"

    audio_folder = "./data/audio/augmented-audios/"
    #model = WhisperModel("small", device="cpu")
    openai_model = OpenAI(api_key=openai_api_key, base_url=base_url)
    for file in os.listdir(audio_folder):

        audio_path = os.path.join(audio_folder, file)

        """ segments, info = model.transcribe(audio_path, beam_size=5)
        print(f"Lingua trovata {info.language} con probabilità {info.language_probability}")
        transcript = ""

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            transcript += segment.text """
        
        with open(audio_path, "rb") as f:
            transcript = openai_model.audio.transcriptions.create(
                file=f,
                model="openai/whisper-large-v3",
                response_format="json",
                temperature=0.0,
                language="en"
            ) 

        print(f"Risultato finale:\n\n{transcript}\n\n")
        break
    