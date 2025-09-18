from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCaseParams
from litellm import completion

class GeminiCorrectnessMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case, *args, **kwargs):
        # Call Gemini API using litellm
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are an evaluator that compares outputs."},
                {"role": "user", "content": f"""
                Input: {test_case.input}
                Expected: {test_case.expected_output}
                Actual: {test_case.actual_output}
                Evaluate correctness: return a score 0 to 1.
                """}
            ]
        )

        # Extract score from Gemini's response (assume numeric output)
        score = float(response['choices'][0]['message']['content'].strip())
        return score

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
