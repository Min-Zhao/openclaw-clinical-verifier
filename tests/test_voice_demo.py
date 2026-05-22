from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from voice_demo import HealthcareVoicePipeline


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> str:
        assert audio_path.name == "question.wav"
        return "I have a fever and chest pain. What should I do?"


class FakeReasoner:
    def __init__(self):
        self.question = ""

    def answer(self, question: str) -> str:
        self.question = question
        return "Chest pain can be urgent. Seek emergency care now."


class FakeSpeaker:
    def __init__(self):
        self.text = ""

    def speak(self, text: str, output_path: Path | None = None) -> Path | None:
        self.text = text
        return output_path


class VoicePipelineTest(unittest.TestCase):
    def test_voice_pipeline_passes_transcript_through_llm_and_tts(self):
        reasoner = FakeReasoner()
        speaker = FakeSpeaker()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "answer.wav"
            pipeline = HealthcareVoicePipeline(FakeTranscriber(), reasoner, speaker)

            result = pipeline.run(Path("question.wav"), output_path)

        self.assertEqual(result.transcript, "I have a fever and chest pain. What should I do?")
        self.assertEqual(reasoner.question, result.transcript)
        self.assertEqual(result.answer, "Chest pain can be urgent. Seek emergency care now.")
        self.assertEqual(speaker.text, result.answer)
        self.assertEqual(result.spoken_output, output_path)


if __name__ == "__main__":
    unittest.main()
