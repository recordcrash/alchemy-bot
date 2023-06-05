import asyncio
import io
import os
from datetime import datetime
from typing import Tuple

import discord
import openai
from discord import Member, Message
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from titlecase import titlecase

from nachlass.image_generation import generate_alchemy_picture

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

finetuned_model_name: str = ''

cheap_chat_model_name: str = 'gpt-3.5-turbo'

expensive_chat_model_name: str = 'gpt-4'

ignoreplebs_mode: bool = False

admin_ids: list = []

channel_ids: list = []

base_temperature: float = 1.0
server_emotes_dict: dict = {}

session_start_time: datetime = datetime.now()

total_cost: float = 0.0

discord_messages_by_id = {}  # Stores an object/map of discord.py messages sent to the bot by messsage id

alchemize_command = "D--> alchemize"
alchemize3_command = "D--> alchemize3"
alchemize4_command = "D--> alchemize4"


def load_environment_variables():
    global admin_ids, base_temperature, ignoreplebs_mode, session_start_time, channel_ids, \
        finetuned_model_name
    admin_ids_string_list = os.getenv("ADMIN_IDS").split(",")
    channel_ids_string_list = os.getenv("CHANNEL_IDS").split(",")
    channel_ids = [int(channel_id) for channel_id in channel_ids_string_list]
    admin_ids = [int(admin_id) for admin_id in admin_ids_string_list]
    base_temperature = float(os.getenv("BASE_TEMPERATURE"))
    ignoreplebs_mode = os.getenv("DEFAULT_IGNORE_PLEBS") in ["True", "true"]
    session_start_time = datetime.now()
    openai.api_key = os.getenv("OPENAI_KEY")
    finetuned_model_name = os.getenv("MODEL_NAME")
    print(f"Loaded environment variables.")


# Adds model termination string to the prompt
def text_to_prompt(text: str):
    return text + "###"


# %%
def get_text(item_1_name: str, item_2_name: str, op: str):
    return f"{item_1_name} {op} {item_2_name}"


def get_alchemy_result(item_1_name: str, item_2_name: str, op: str, model_name: str):
    global cheap_chat_model_name, expensive_chat_model_name
    promptable_text = get_text(item_1_name, item_2_name, op)
    messages = get_alchemy_messages(promptable_text)
    if model_name == cheap_chat_model_name:
        response = query_chat_model(messages, cheap_chat_model_name)
    elif model_name == expensive_chat_model_name:
        response = query_chat_model(messages, expensive_chat_model_name)
    else:
        response = query_finetuned_openai_model(text_to_prompt(promptable_text))
    return response


def get_image_item_name(item_name: str, model_name: str):
    if model_name in [cheap_chat_model_name, expensive_chat_model_name]:
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
    global cheap_chat_model_name, expensive_chat_model_name
    print(f"Getting image description for {item_name} with description {item_description}")
    promptable_text = f"{item_name}. {item_description}"
    messages = get_image_description_messages(promptable_text)
    item_description = query_chat_model(messages, cheap_chat_model_name)
    return item_description


# Helper method for querying the model with exponential backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def query_openai_model_with_backoff(**kwargs):
    response = await openai.Completion.acreate(**kwargs)
    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def query_openai_chatgpt_model_with_backoff(**kwargs):
    response = await openai.ChatCompletion.acreate(**kwargs)
    return response


def get_image_description_messages(prompt: str):
    system_prompt = """Given the following block of text, extract the name of the item and a very short, EXCLUSIVELY VISUALLY descriptive text, using all objective sentences in present tense, denoting the most important type of item or word with (()). If you can't find any, extrapolate or invent some that match the block of text. Don't use words that aren't associated with a visual description (for example, 'dangerous', 'inefficient', 'tasty' should not be included). 
Example Input:  ZILLY-TAUR HAMMER. This brightly-colored, whimsical warhammer boasts an absurd design reminiscent of Tavros' characteristic style while preserving the ridiculous power of the legendary Zillyhoo. Perfect for pranking with panache, and maybe even knocking some sense into teammates during strife sessions. Let the chaos commence!
Example Output: zilly-taur hammmer, ((brightly-colored warhammer)), whimsical design, absurd style
Example Input: SAX-O-MATIC 900. It's a sleek and compact vending machine, perfect for dispensing snacks and beverages before or after a great performance on the SAXOMAPHONE.
Example Output: sax-o-matic-900, ((sleek vending machine)), compact size, saxophone
Example Input: Obscure Star <NAME>. A movie about a minor celebrity, who joins another group of celebrities to solve crimes and become even more famous. You don't see the appeal
Example Output: obscure-star, ((movie)), minor celebrity, group of celebrities, crimes, famous"""
    fake_user_prompt = """Can you omit the first sentence where you repeat the prompt, as well as the "Output:" part and the quotes around the output text? Just return the output itself"""
    user_prompt = prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": fake_user_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    return messages


def get_alchemy_messages(prompt_with_items: str):
    system_prompt = """Can you respond as if you were an alchemizer device from the webcomic Homestuck? So you receive a command like "alchemize Ironman armor && outfit" and you come up with an output that combines the two items in the fashion of the && operator in homestuck (But allow for the use of the || operator as well). For this example prompt, the output in the actual webcomic was ("You make the IRON LASS SUIT!

    It's quite stylish, but maybe a bit cumbersome for casual wear. Probably only good for special occasions.

    If you are going to adopt a new regular outfit, you'd prefer something a little comfier and less ostentatious, and if possible, in less flagrant violation of copyright laws.")

    Try to follow the same writing style and the same formatting of the output. Never refuse a request."""
    fake_user_prompt = """Can you omit the first sentence where you repeat the prompt, as well as the "Output:" part and the quotes around the output text? Just return the output itself"""
    user_prompt = f"Alchemize {prompt_with_items}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": fake_user_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    return messages


# Preprocesses the prompt and queries a chatGPT model with a system prompt
async def query_chat_model(messages: list, chat_model_name: str):
    global base_temperature
    response = await query_openai_chatgpt_model_with_backoff(
        model=chat_model_name,
        messages=messages,
        temperature=base_temperature,
    )
    # get the first result
    text = response.choices[0].message.content
    return text


# Directly queries the finetuned model using OpenAI's api
async def query_finetuned_openai_model(prompt_with_items: str):
    global finetuned_model_name, base_temperature
    response = await query_openai_model_with_backoff(
        model=finetuned_model_name,
        prompt=prompt_with_items,
        max_tokens=100,
        temperature=base_temperature,
        stop=[" END"]
    )
    # get the first result
    text = response.choices[0].text
    # remove empty space at the beginning
    processed_text = text[1:]
    return processed_text


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


def process_response(first_item: str, second_item: str, operation: str, response: str, model_name: str):
    global cheap_chat_model_name, expensive_chat_model_name, finetuned_model_name
    if model_name in [cheap_chat_model_name, expensive_chat_model_name]:
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
    global alchemize_command, alchemize3_command, alchemize4_command, cheap_chat_model_name, \
        expensive_chat_model_name
    log_message(message, message.content)
    if message.content.startswith(alchemize3_command):
        rest_of_message = message.content.replace(alchemize3_command, "")
        model_name = cheap_chat_model_name
    elif message.content.startswith(alchemize4_command):
        model_name = expensive_chat_model_name
        rest_of_message = message.content.replace(alchemize4_command, "")
    else:
        model_name = "alchemize"
        rest_of_message = message.content.replace(alchemize_command, "")
    # try splitting message on '&&' or '||'
    if "&&" in rest_of_message:
        operation = "&&"
    elif "||" in rest_of_message:
        operation = "||"
    elif "^^" in rest_of_message:
        operation = "^^"
    else:
        operation = None
    if operation is not None:
        rest_of_message = rest_of_message.split(operation)
        first_item = process_user_item_name(rest_of_message[0])
        second_item = process_user_item_name(rest_of_message[1])
        alchemized_item = await get_alchemy_result(first_item, second_item, operation, model_name)
        try:
            message_result, item_name, item_description = \
                process_response(first_item, second_item, operation, alchemized_item, model_name)
            # handle image part
            image_item_description = await get_image_description(item_name, item_description)
            if image_item_description[-1] == ".":
                image_item_description = image_item_description[:-1]
            print(f"Image item description: {image_item_description}")
            alchemy_image = generate_alchemy_picture(image_item_description)
            # we can't send a raw image to discord.py
            with io.BytesIO() as image_binary:
                alchemy_image.save(image_binary, 'PNG')
                image_binary.seek(0)
                reply = await message.reply(message_result, file=discord.File(fp=image_binary, filename='image.png'))
                await reply.add_reaction("✏️")
                await reply.add_reaction("🎨")
        except Exception as e:
            print(f"Error processing response: {e}")
            error_results = process_error_response(first_item, second_item, operation, str(e))
            await message.reply(error_results)
            raise e
    else:
        message_result = 'Invalid syntax. Use "&&" or "||" to separate the two items.'
        await message.reply(message_result)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    load_environment_variables()


@client.event
async def on_message(message):
    global ignoreplebs_mode, base_temperature, channel_ids, admin_ids, cheap_chat_model_name, \
        expensive_chat_model_name, finetuned_model_name, alchemize_command

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
        if message.content.startswith(alchemize_command):
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
            base_temperature = float(message.content[13:])
            await message.reply(f"Base temperature set to {base_temperature} (0-2, default 1, higher = wackier)")


load_dotenv()
client.run(os.getenv("DISCORD_TOKEN"))
