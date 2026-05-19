from supertonic import TTS

def main():
    print("Hello from supertonic-tts!")

    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name="M1")

    text = "आज के समय में तकनीक हमारे जीवन का एक अहम हिस्सा बन चुकी है। सुबह उठने से लेकर रात को सोने तक, हम किसी न किसी रूप में तकनीक का उपयोग करते हैं। स्मार्टफोन, इंटरनेट और कंप्यूटर ने हमारे काम करने के तरीके को पूरी तरह बदल दिया है।"
    wav, duration = tts.synthesize(text, voice_style=style, lang="en")

    tts.save_audio(wav, "output-hin.wav")
    print(f"Generated {duration}s of audio")



if __name__ == "__main__":
    main()
