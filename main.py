import sys
import time
from typing import Generator
from supertonic import TTS
from test_streaming import SupertonicStreamingPipeline


def stream_text_output(text: str, delay: float = 0.05) -> Generator[str, None, None]:
    """
    Streams the input text by yielding word-level tokens (with spaces)
    and outputting them to stdout in real-time with a simulated delay.

    Args:
        text (str): The input text to stream.
        delay (float): Delay in seconds between each yielded token.

    Yields:
        str: Individual text tokens.
    """
    words = text.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        try:
            sys.stdout.write(token)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Fallback for terminals with limited encoding support (e.g. Windows CP1252)
            sys.stdout.write(token.encode("ascii", "backslashreplace").decode("ascii"))
            sys.stdout.flush()
        time.sleep(delay)
        yield token


def main():
    # Attempt to reconfigure stdout encoding to UTF-8 to handle Unicode characters seamlessly
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== Supertonic-TTS Streaming Demo ===\n")

    text = "आज के समय में तकनीक हमारे जीवन का एक अहम हिस्सा बन चुकी है। सुबह उठने से लेकर रात को सोने तक, हम किसी न किसी रूप में तकनीक का उपयोग करते हैं। स्मार्टफोन, इंटरनेट और कंप्यूटर ने हमारे काम करने के तरीके को पूरी तरह बदल दिया है।"

    print("--- 1. Simulated Real-Time Text Stream ---")
    # Wrap in a list or consume it so we can print the streaming effect
    token_generator = stream_text_output(text, delay=0.04)
    tokens = list(token_generator)
    print("\n\n--- 2. End-to-End Text to PCM Audio Streaming ---")

    # Initialize the streaming audio pipeline
    pipeline = SupertonicStreamingPipeline(voice_name="M1", lang="hi")

    # Generate the text stream again for synthesis
    text_stream = stream_text_output(text, delay=0.01)

    print("\nSynthesizing streaming audio...")
    pcm_chunks = []
    chunk_count = 0

    # Pass the text stream generator directly into the pipeline
    for pcm_chunk in pipeline.stream_text_to_pcm(text_stream, min_char_threshold=60):
        chunk_count += 1
        pcm_chunks.append(pcm_chunk)
        print(
            f"\n[Audio Chunk {chunk_count}: Generated {len(pcm_chunk)} PCM bytes]",
            end="",
            flush=True,
        )

    print("\n\nStreaming synthesis complete.")
    all_pcm = b"".join(pcm_chunks)
    print(f"Total chunks generated: {chunk_count}")
    print(f"Total PCM bytes accumulated: {len(all_pcm)}")


if __name__ == "__main__":
    main()
