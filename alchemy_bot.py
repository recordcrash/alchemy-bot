import asyncio
import io
import json
import os

import discord
from openai import AsyncOpenAI as OpenAI

from dotenv import load_dotenv
from titlecase import titlecase

from image_generation import generate_alchemy_picture
from text_generation import BASE_TEMPERATURE, FINETUNED_MODEL_NAME, CHEAP_CHAT_MODEL_NAME, EXPENSIVE_CHAT_MODEL_NAME, \
    request_text_to_prompt, get_messages_from_system_and_user_prompts, \
    query_chat_model, query_finetuned_openai_model

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

ignoreplebs_mode: bool = False

admin_ids: list = []

channel_ids: list = []

temperature: float = 1.0

openai_client = None

discord_messages_by_id = {}  # Stores an object/map of discord.py messages sent to the bot by messsage id

ALCHEMIZE_COMMAND = "D--> alchemize"
ALCHEMIZE3_COMMAND = "D--> alchemize3"
ALCHEMIZE4_COMMAND = "D--> alchemize4"

IMAGE_DESCRIPTION_SYSTEM_PROMPT = open("prompts/image_description_system_prompt.txt", "r").read()
IMAGE_DESCRIPTION_FAKE_USER_PROMPT = open("prompts/image_description_fake_user_prompt.txt", "r").read()

# ALCHEMY_SYSTEM_PROMPT = open("prompts/alchemy_system_prompt.txt", "r").read()
# ALCHEMY_FAKE_USER_PROMPT = open("prompts/alchemy_fake_user_prompt.txt", "r").read()
ALCHEMY_SYSTEM_PROMPT = open("prompts/alchemy_system_prompt_json.txt", "r").read()
ALCHEMY_FAKE_USER_PROMPT = open("prompts/alchemy_fake_user_prompt_json.txt", "r").read()


def load_environment_variables():
    global admin_ids, temperature, ignoreplebs_mode, channel_ids, openai_client
    admin_ids_string_list = os.getenv("ADMIN_IDS").split(",")
    channel_ids_string_list = os.getenv("CHANNEL_IDS").split(",")
    channel_ids = [int(channel_id) for channel_id in channel_ids_string_list]
    admin_ids = [int(admin_id) for admin_id in admin_ids_string_list]
    temperature = BASE_TEMPERATURE
    openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    ignoreplebs_mode = os.getenv("DEFAULT_IGNORE_PLEBS") in ["True", "true"]
    print(f"Loaded environment variables.")


def get_alchemy_result(item_1_name: str, item_2_name: str, op: str, model_name: str):
    global temperature, openai_client
    promptable_text = f"{item_1_name} {op} {item_2_name}"
    messages = get_alchemy_messages(promptable_text)
    if model_name == CHEAP_CHAT_MODEL_NAME:
        response = query_chat_model(openai_client, messages, CHEAP_CHAT_MODEL_NAME, temperature=temperature)
    elif model_name == EXPENSIVE_CHAT_MODEL_NAME:
        response = query_chat_model(openai_client, messages, EXPENSIVE_CHAT_MODEL_NAME, temperature=temperature)
    else:
        response = query_finetuned_openai_model(openai_client, request_text_to_prompt(promptable_text), temperature=temperature)
    return response


def get_image_item_name(item_name: str, model_name: str):
    if model_name in [CHEAP_CHAT_MODEL_NAME, EXPENSIVE_CHAT_MODEL_NAME]:
        # chat item names tend to come in the form You create the ITEM NAME! or YOU GET THE ITEM NAME!,
        # we want to remove the "You create the" part and
        # any punctuation at the end
        # we can probably get away with just taking the part in all caps and removing any exclamations if they exist
        # we'll look for the first ! and then take the part before that
        if "!" in item_name:
            item_name = item_name.split("!")[0]
            # go backwards until we find the first lowercase letter
            for i in range(len(item_name) - 1, -1, -1):
                if item_name[i].islower():
                    item_name = item_name[i + 1:]
                    break
        # remove any variations of "you get", "you create" or "..."
        # case insensitive
        print(f"Item name before: {item_name}")
        item_name = item_name.replace("you get", "").replace("you create", "")
        item_name = item_name.replace("You get", "").replace("You create", "")
        item_name = item_name.replace("YOU GET", "").replace("YOU CREATE", "")
        item_name = item_name.replace("...", "").strip()
        print(f"Item name after: {item_name}")
    return item_name


def get_image_description(item_name: str, item_description: str):
    global temperature, openai_client
    print(f"Getting image description for {item_name} with description {item_description}")
    promptable_text = f"{item_name}. {item_description}"
    messages = get_image_description_messages(promptable_text)
    item_description = query_chat_model(openai_client, messages, CHEAP_CHAT_MODEL_NAME, temperature=temperature)
    return item_description


def get_image_description_messages(prompt: str):
    messages = get_messages_from_system_and_user_prompts(system_prompts=[IMAGE_DESCRIPTION_SYSTEM_PROMPT],
                                                         user_prompts=[IMAGE_DESCRIPTION_FAKE_USER_PROMPT, prompt])
    return messages


def get_alchemy_messages(prompt_with_items: str):
    user_prompt = f"Alchemize {prompt_with_items}"
    messages = get_messages_from_system_and_user_prompts(system_prompts=[ALCHEMY_SYSTEM_PROMPT],
                                                         user_prompts=[ALCHEMY_FAKE_USER_PROMPT, user_prompt])
    return messages


# Helper logger method for tracking bot usage
def log_message(message, text):
    print(f"Message author id: {message.author.id}, "
          f"nickname: {message.author.nick if message.author.nick is not None else message.author.name}, "
          f"content: {text}")


def process_user_item_name(item_name: str):
    item_name = item_name.strip()
    # if not all caps
    if not item_name.isupper():
        item_name = titlecase(item_name)
    return item_name


def process_error_response(first_item: str, second_item: str, operation: str, error: str):
    message_result = f"You alchemize *{first_item}* {operation} *{second_item}*. You get...\n" \
                     f"## NOTHING!\n" \
                     f"**Because there was an error processing the request.**\n" \
                     f"*Error message:* {error}\n"
    return message_result


async def process_response_json(first_item: str, second_item: str, operation: str, response: str, model_name: str):
    if model_name in [CHEAP_CHAT_MODEL_NAME, EXPENSIVE_CHAT_MODEL_NAME]:
        # response will come in json format, with "name", "description", "visual_prompt"
        # so let's parse the response
        print(f"JSON Response from combination: {response}")
        response = json.loads(response)
        item_name = response["name"]
        item_description = response["description"]
        message_result = f"You alchemize *{first_item}* {operation} *{second_item}*.\n" \
                         f"## {item_name}\n" \
                         f"**{item_description}**\n"
        visual_prompt = response["visual_prompt"]
    else:
        message_result, item_name, item_description = \
            process_response(first_item, second_item, operation, response, model_name)
        visual_prompt = await get_image_description(item_name, item_description)
    return message_result, item_name, item_description, visual_prompt

def process_response(first_item: str, second_item: str, operation: str, response: str, model_name: str):
    if model_name in [CHEAP_CHAT_MODEL_NAME, EXPENSIVE_CHAT_MODEL_NAME]:
        # first line is the name, next lines are the description
        item_name = response.split("\n")[0]
        item_description = "\n".join(response.split("\n")[1:]).strip()
        message_result = f"You alchemize *{first_item}* {operation} *{second_item}*.\n" \
                         f"## {item_name}\n" \
                         f"**{item_description}**\n"
    else:
        # the result is NAME (DESCRIPTION), let's get them in variables
        item_name = response.split(" (")[0]
        item_description = response.split(" (")[1][:-1]
        message_result = f"You alchemize *{first_item}* {operation} *{second_item}*. You get...\n" \
                         f"## {item_name}\n" \
                         f"**{item_description}**\n"
    return message_result, item_name, item_description


async def handle_alchemize(message):
    log_message(message, message.content)
    if message.content.startswith(ALCHEMIZE3_COMMAND):
        rest_of_message = message.content.replace(ALCHEMIZE3_COMMAND, "")
        model_name = CHEAP_CHAT_MODEL_NAME
    elif message.content.startswith(ALCHEMIZE4_COMMAND):
        model_name = EXPENSIVE_CHAT_MODEL_NAME
        rest_of_message = message.content.replace(ALCHEMIZE4_COMMAND, "")
    else:
        model_name = FINETUNED_MODEL_NAME
        rest_of_message = message.content.replace(ALCHEMIZE_COMMAND, "")
    # try splitting message on operation
    if "&&" in rest_of_message:
        operation = "&&"
    elif "||" in rest_of_message:
        operation = "||"
    elif "^^" in rest_of_message:
        operation = "^^"
    elif "!&" in rest_of_message:
        operation = "!&"
    else:
        operation = None
    if operation is not None:
        rest_of_message = rest_of_message.split(operation)
        first_item = process_user_item_name(rest_of_message[0])
        second_item = process_user_item_name(rest_of_message[1])
        alchemized_item = await get_alchemy_result(first_item, second_item, operation, model_name)
        try:
            message_result, item_name, item_description, visual_prompt = \
                await process_response_json(first_item, second_item, operation, alchemized_item, model_name)
            # handle image part
            print(f"Image item description: {visual_prompt}")
            alchemy_image = await generate_alchemy_picture(openai_client, visual_prompt)
            # we can't send a raw image to discord.py
            with io.BytesIO() as image_binary:
                alchemy_image.save(image_binary, 'PNG')
                image_binary.seek(0)
                reply = await message.reply(message_result, file=discord.File(fp=image_binary, filename='image.png'))
                await reply.add_reaction("✏️")
                await reply.add_reaction("🎨")
        except Exception as e:
            print(f"Error processing response: {e} traceback: {e.__traceback__}")
            error_results = process_error_response(first_item, second_item, operation, str(e))
            await message.reply(error_results)
            raise e
    else:
        message_result = 'Invalid syntax. Use "&&", "||", "^^" or "!&" to separate the two items.'
        await message.reply(message_result)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    load_environment_variables()


@client.event
async def on_message(message):
    global ignoreplebs_mode, temperature, channel_ids, admin_ids

    # don't respond to bots
    if message.author.bot:
        return

    # don't respond if the channel_ids is not an empty array and has ids that are not the current one
    if len(channel_ids) > 0 and message.channel.id not in channel_ids:
        return

    # don't respond to ourselves
    if message.author == client.user:
        return

    if not ignoreplebs_mode:
        if message.content.startswith(ALCHEMIZE_COMMAND):
            asyncio.create_task(handle_alchemize(message))

    # only allow admins to use these commands
    if message.author.id in admin_ids:
        if message.content.startswith('D--> ignoreplebsier'):
            ignoreplebs_mode = True
            await message.reply("Ignoreplebs mode activated")

        if message.content.startswith('D--> stopignoreplebsier'):
            ignoreplebs_mode = False
            await message.reply("Ignoreplebs mode deactivated")

        if message.content.startswith('D--> setheat'):
            temperature = float(message.content[13:])
            await message.reply(f"Base temperature set to {temperature} (0-2, default 1, higher = wackier)")


load_dotenv()


client.run(os.getenv("DISCORD_TOKEN"))
#
# def test_system():
#
#     first_item = "Nunchucks"
#     second_item = "Picture of a Fat Husky"
#     operation = "&&"
#     model_name = CHEAP_CHAT_MODEL_NAME
#     alchemy = get_alchemy_result(first_item, second_item, operation, model_name)
#     return alchemy
#
# load_environment_variables()
# alchemy_result = asyncio.run(test_system())
# print(alchemy_result)
