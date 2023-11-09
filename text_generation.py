# Helper class for text generation, contacts OpenAI's API
# three models will be available:
# 1. A finetuned davinci model
# 2. GPT-3.5-turbo
# 3. GPT-4

import os

from openai import OpenAI

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

load_dotenv()

FINETUNED_MODEL_NAME = os.getenv("MODEL_NAME")

BASE_TEMPERATURE = float(os.getenv("BASE_TEMPERATURE"))

CHEAP_CHAT_MODEL_NAME = 'gpt-3.5-turbo'

EXPENSIVE_CHAT_MODEL_NAME = 'gpt-4-1106-preview'
# EXPENSIVE_CHAT_MODEL_NAME = 'gpt-4'


# Queries a chatGPT model with messages using OpenAI's api
async def query_chat_model(client, messages: list, chat_model_name: str, temperature: float = BASE_TEMPERATURE):
    if chat_model_name == 'gpt-4-1106-preview':
        response = await query_openai_chatgpt_model_with_backoff(
            client=client,
            model=chat_model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
    else:
        response = await query_openai_chatgpt_model_with_backoff(
            client=client,
            model=chat_model_name,
            messages=messages,
            temperature=temperature,
        )
    # get the first result
    text = response.choices[0].message.content
    return text


# Directly queries the finetuned model using OpenAI's api
async def query_finetuned_openai_model(client, prompt_with_items: str, temperature: float = BASE_TEMPERATURE):
    response = await query_openai_model_with_backoff(
        client=client,
        model=FINETUNED_MODEL_NAME,
        prompt=prompt_with_items,
        max_tokens=500,
        temperature=temperature,
        stop=[" END"]
    )
    # get the first result
    text = response.choices[0].text
    # remove empty space at the beginning
    processed_text = text[1:]
    return processed_text


# API methods
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def query_openai_model_with_backoff(client, **kwargs):
    response = await client.completions.create(**kwargs)
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def query_openai_chatgpt_model_with_backoff(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)
    return response


def get_messages_from_system_and_user_prompts(system_prompts: list, user_prompts: list):
    messages = []
    for i in range(len(system_prompts)):
        messages.append({
            "role": "system",
            "content": system_prompts[i]
        })
    for i in range(len(user_prompts)):
        messages.append({
            "role": "user",
            "content": user_prompts[i]
        })
    return messages


# Adds model termination string to the prompt
def request_text_to_prompt(text: str):
    return text + "###"
