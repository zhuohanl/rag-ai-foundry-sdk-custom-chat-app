import os
import sys
import pathlib
import pandas as pd
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
from azure.ai.evaluation import evaluate, GroundednessEvaluator, CoherenceEvaluator, FluencyEvaluator, RelevanceEvaluator
from azure.identity import DefaultAzureCredential

# Add the src directory to Python path
src_path = pathlib.Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from chat_with_products import chat_with_products

# load environment variables from the .env file at the root of this repo
from dotenv import load_dotenv

load_dotenv()

# create a project client using environment variables loaded from the .env file
project = AIProjectClient.from_connection_string(
    conn_str=os.environ["AIPROJECT_CONNECTION_STRING"], credential=DefaultAzureCredential()
)

connection = project.connections.get_default(connection_type=ConnectionType.AZURE_OPEN_AI, include_credentials=True)

evaluator_model = {
    "azure_endpoint": connection.endpoint_url,
    "azure_deployment": os.environ["EVALUATION_MODEL"],
    "api_version": "2024-06-01",
    "api_key": connection.key,
}

# Initialize all evaluators
groundedness = GroundednessEvaluator(evaluator_model)
coherence = CoherenceEvaluator(evaluator_model)
fluency = FluencyEvaluator(evaluator_model)
relevance = RelevanceEvaluator(evaluator_model)


#########################################################################
# Create a wrapper function that implements the evaluation interface
#########################################################################

def evaluate_chat_with_products(query):
    response = chat_with_products(messages=[{"role": "user", "content": query}])
    return {"response": response["message"].content, "context": response["context"]["grounding_data"]}


#########################################################################
# Run the evaluation
#########################################################################

# Evaluate must be called inside of __main__, not on import
if __name__ == "__main__":
    from config import ASSET_PATH, EVAL_OUTPUT_PATH

    # Create the output directory if it doesn't exist
    os.makedirs(EVAL_OUTPUT_PATH, exist_ok=True)

    # workaround for multiprocessing issue on linux
    from pprint import pprint
    from pathlib import Path
    import multiprocessing
    import contextlib

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

    # run evaluation with a dataset and target function, log to the project
    result = evaluate(
        data=Path(ASSET_PATH) / "chat_eval_data.jsonl",
        target=evaluate_chat_with_products,
        evaluation_name="evaluate_chat_with_products",
        evaluators={
            "groundedness": groundedness,
            "coherence": coherence,
            "fluency": fluency,
            "relevance": relevance,
        },
        evaluator_config={
            "default": {
                "query": {"${data.query}"},
                "response": {"${target.response}"},
                "context": {"${target.context}"},
            }
        },
        azure_ai_project=project.scope,
        output_path=Path(EVAL_OUTPUT_PATH) / "myevalresults.json",
    )

    tabular_result = pd.DataFrame(result.get("rows"))

    pprint("-----Summarized Metrics-----")
    pprint(result["metrics"])
    pprint("-----Tabular Result-----")
    pprint(tabular_result)
    pprint(f"View evaluation results in AI Studio: {result['studio_url']}")