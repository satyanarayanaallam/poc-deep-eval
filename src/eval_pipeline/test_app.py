from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.llms import GeminiModel
import os
from dotenv import load_dotenv

# Set your Google API key
load_dotenv()

# Now access your API key
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
gemini = GeminiModel(model_name="gemini-2.0-flash", api_key=api_key)

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=gemini   # 👈 tell deepeval to use Gemini instead of GPT
)

similarity_metric = GEval(
    name="SemanticSimilarity",
    criteria="Evaluate if the actual output has the same meaning as the expected output, even if worded differently.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7,
    model=gemini
)

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
    expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
)

evaluate([test_case], [correctness_metric,similarity_metric])