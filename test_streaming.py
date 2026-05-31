import re
import numpy as np
from typing import Generator, Iterator
from supertonic import TTS


class SupertonicStreamingPipeline:
    def __init__(
        self, voice_name: str = "M1", lang: str = "en", auto_download: bool = True
    ):
        """
        Initializes the streaming pipeline by loading the Supertonic-3 ONNX model
        and caching the selected voice style.
        """
        self.tts = TTS(auto_download=auto_download)
        self.style = self.tts.get_voice_style(voice_name=voice_name)
        self.lang = lang
        self.sample_rate = 44100  # Default Supertonic-3 sample rate [5, 7]

        # Splitting regex targeting standard punctuation boundaries (.!? , ; \n),
        # utilizing a negative lookahead assertion to prevent slicing inside tag brackets <...>
        self.boundary_regex = re.compile(r"(?<=[.!?,\n;])\s+(?![^<]*>)")

    def stream_text_to_pcm(
        self,
        token_stream: Iterator[str],
        min_char_threshold: int = 120,
        total_steps: int = 8,
        speed: float = 1.05,
    ) -> Generator[bytes, None, None]:
        """
        Consumes an incoming sequence of text tokens, buffers them dynamically,
        and yields sequential chunks of raw 16-bit Mono PCM bytes.
        """
        buffer = ""
        for token in token_stream:
            buffer += token

            # Identify logical syntactic boundaries in the accumulated text
            boundaries = list(self.boundary_regex.finditer(buffer))
            if not boundaries:
                continue

            # Select the split index that best matches the target character threshold
            split_idx = -1
            for match in boundaries:
                if match.start() >= min_char_threshold:
                    split_idx = match.start()
                    break

            # Force-slice at the last known punctuation boundary if the buffer grows too large
            if split_idx == -1 and len(buffer) > min_char_threshold * 2:
                split_idx = boundaries[-1].start()

            if split_idx != -1:
                chunk_text = buffer[: split_idx + 1].strip()
                buffer = buffer[split_idx + 1 :]

                if chunk_text:
                    pcm_bytes = self._synthesize_chunk_to_pcm(
                        chunk_text, total_steps, speed
                    )
                    if pcm_bytes:
                        yield pcm_bytes

        # Flush any remaining text at the end of the stream
        remaining_text = buffer.strip()
        if remaining_text:
            pcm_bytes = self._synthesize_chunk_to_pcm(
                remaining_text, total_steps, speed
            )
            if pcm_bytes:
                yield pcm_bytes

    def _synthesize_chunk_to_pcm(
        self, text: str, total_steps: int, speed: float
    ) -> bytes:
        """
        Executes on-device ONNX inference on a discrete chunk of text
        and converts the output to 16-bit signed PCM format.
        """
        try:
            # Execute core on-device synthesis
            wav, _ = self.tts.synthesize(
                text=text,
                lang=self.lang,
                voice_style=self.style,
                total_steps=total_steps,
                speed=speed,
            )

            # Output is a 2D numpy array of shape (1, samples) with float32 values [1, 15]
            audio_data = wav.squeeze()

            # Prevent clipping by capping the amplitude range
            audio_clipped = np.clip(audio_data, -1.0, 1.0)

            # Translate floating-point values to standard 16-bit integers
            pcm_16 = (audio_clipped * 32767.0).astype(np.int16)

            return pcm_16.tobytes()
        except Exception as error:
            print(f"ONNX synthesis step execution error: {str(error)}")
            return b""
