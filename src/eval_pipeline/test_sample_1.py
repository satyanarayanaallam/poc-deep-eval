from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from dotenv import load_dotenv
from deepeval.models.llms import GeminiModel
import os

# Load env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
gemini = GeminiModel(model_name="gemini-2.0-flash", api_key=api_key)

def test_sample_1():
    correctness_metric = GEval(
        name="correctness",
        criteria="Does the response correctly answer the question based on the provided context?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=gemini
    )

    # Input data (the question to the model)
    input_data = {"question": "What is 2 + 2?"}
    
    # Get the actual output from the model
    actual_output = gemini.generate(input_data["question"])
    print(f"Actual Output: {actual_output}")
    
    # Build test case correctly
    test_case = LLMTestCase(
        name="Sample Test Case 1",
        description="Test case to evaluate the correctness of a simple addition operation.",
        input=input_data["question"],   # 👈 must use input_data, not input
        expected_output="4",
        actual_output=actual_output[0].strip(),
        metrics=[correctness_metric]
    )

    assert_test(test_case, [correctness_metric])
