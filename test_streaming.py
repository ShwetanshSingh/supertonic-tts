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

    def _find_split_index(self, buffer: str, min_char_threshold: int) -> int:
        """
        Identifies the optimal split index in the accumulated buffer
        based on punctuation boundaries and character thresholds.
        """
        boundaries = list(self.boundary_regex.finditer(buffer))
        if not boundaries:
            return -1

        # Find the first boundary that exceeds the min_char_threshold
        for match in boundaries:
            if match.start() >= min_char_threshold:
                return match.start()

        # Force-slice at the last known punctuation boundary if the buffer is too large
        if len(buffer) > min_char_threshold * 2:
            return boundaries[-1].start()

        return -1

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

            split_idx = self._find_split_index(buffer, min_char_threshold)
            if split_idx == -1:
                continue

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
