from typing import List, Tuple
from autogen_core.models import CreateResult, LLMMessage, SystemMessage
from src.copilot.chat_client.chat_client_creator import ChatClientCreator
from src.configurator.configurator import Configurator
from src.connector.connector import Connector

async def evaluate_by_llm(prediction: str, ans: str) -> Tuple[bool, str]:
    """
    Evaluates the correctness of a prediction using an LLM.
    
    Args:
        prediction (str): The AI's prediction.
        ans (str): The expected answer.
    
    Returns:
        Tuple[bool, str]: (is_correct, llm_result)
    """
    creator = ChatClientCreator(
        Connector.get_keycloak(),
        endpoint=Configurator.get_config()["llm_gateway"]["host"]
    )
    chat_client = creator.create() 
    messages: List[LLMMessage] = [ 
        SystemMessage(
            content=(
                "Role:\n"
                "You are an AI judge that evaluates the correctness of a prediction. "
                "Another AI has made a prediction about the number of shipments that can be retrieved based on a given input. "
                "Your task is to determine if the prediction is correct or not.\n"
                "Instructions:\n"
                "- You will be given a prediction and the expected answer.\n"
                "- Your task is to determine if the prediction is correct based on the expected answer.\n"
                '- Use <is_correct="[True/False]"> to indicate the correctness of the prediction.\n'
                "- Also provide a brief explanation of your reasoning.\n"
                "- Please note that the number of shipments retrieved does not matter, as long as the prediction is correct.\n"
                "- A None is always considered as an incorrect prediction.\n"
                f"Prediction details:\n"
                f"- The prediction (content): {prediction}\n"
                f"- The expected answer (content): {ans}\n"
            ),
        )
    ]
    result: CreateResult = await chat_client.create(messages)
    llm_result = result.content
    is_correct = '<is_correct="True">' in llm_result
    return is_correct, llm_result