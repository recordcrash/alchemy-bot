# Helper class for image generation, contacts local AUTOMATIC1111 stable diffusion API
# location is usually http://127.0.0.1:7860/api/
import base64
import io
from typing import Tuple

import requests
from PIL import Image, ImageOps, ImageDraw

# in order to get this to work, just follow instructions at https://github.com/AUTOMATIC1111

SERVER_IP = "http://127.0.0.1:7860/"

TEXT2IMG_ENDPOINT = SERVER_IP + "sdapi/v1/txt2img/"

IMG2IMG_ENDPOINT = SERVER_IP + "sdapi/v1/img2img/"

OPTIONS_ENDPOINT = SERVER_IP + "sdapi/v1/options/"

REMBG_ENDPOINT = SERVER_IP + "rembg/"

DEFAULT_REMBG_PAYLOAD = {
    "input_image": "",
    "model": "sam",
    "return_mask": False,
    "alpha_matting": False,
    "alpha_matting_foreground_threshold": 240,
    "alpha_matting_background_threshold": 10,
    "alpha_matting_erode_size": 10
}

DEFAULT_IMG2IMG_PAYLOAD = {
    "init_images": [],
    "denoising_strength": 0.75,
    "image_cfg_scale": 0,
    "mask_blur": 4,
    "inpainting_fill": 0,
    "inpaint_full_res": False,
    "prompt": "",
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "batch_size": 1,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
}

FANTASSIFIED_ICONS_CHECKPOINT = "fantassifiedIcons_fantassifiedIconsV20.safetensors [8340e74c3e]"

GAME_ICON_INSTITUTE_CHECKPOINT = "gameIconInstitute_v30.safetensors [c112297163]"

STABLE_DIFFUSION_1_5_CHECKPOINT = "v1-5-pruned.ckpt [e1441589a6]"

DEFAULT_TEXT2IMG_PAYLOAD = {
    "override_settings": {
        "sd_model_checkpoint": GAME_ICON_INSTITUTE_CHECKPOINT
    },
    "enable_hr": False,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "prompt": "",
    "styles": [],
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "sampler_name": "DPM++ 2M SDE Karras",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 50,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "negative_prompt": "",
    "eta": 0,
    "s_min_uncond": 0,
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings_restore_afterwards": True,
    "script_args": [],
    "sampler_index": "DPM++ 2M SDE Karras",
    "script_name": None,
    "send_images": True,
    "save_images": True,
    "alwayson_scripts": {}
}

# POSITIVE_PROMPT_SUFFIX = ", ((item)), ((isolated on green background))"
#
# NEGATIVE_PROMPT_SUFFIX = "easynegative, ng_deepnegative_v1_75t, bad_prompt_version2, bad-artist"

# POSITIVE_PROMPT_SUFFIX = ", ((game icon))"
# NEGATIVE_PROMPT_SUFFIX = ", easynegative, ng_deepnegative_v1_75t, bad_prompt_version2, bad-artist"

# we're using natural language for the DALL·E 3 api here
POSITIVE_PROMPT_SUFFIX = "Game inventory asset, uncropped, transparent background."
NEGATIVE_PROMPT_SUFFIX = ""

ALCHEMITER_ASSET = Image.open("images/alchemiter_pad.png").convert("RGBA")


def generate_image(positive_prompt: str, negative_prompt: str) -> Tuple[Image, str]:
    # 1. contact the API with that prompt, DPM++ 2M SDE Karras, and default settings with batch size 1, silueta as the rembg model
    # 2. return the image
    payload = DEFAULT_TEXT2IMG_PAYLOAD.copy()
    payload["prompt"] = positive_prompt
    payload["negative_prompt"] = negative_prompt
    print(f"Contacting txt2img API at {TEXT2IMG_ENDPOINT} with payload")
    response = requests.post(url=TEXT2IMG_ENDPOINT, json=payload)
    response_json = response.json()
    image_base64 = response_json["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(image_base64.split(",", 1)[0])))
    return image, image_base64


async def generate_image_dalle3(client, positive_prompt: str, negative_prompt: str) -> Tuple[Image, str]:
    # 1. using openAI's python api, contact the API with the prompt
    # 2. return the image
    response = await client.images.generate(
        model="dall-e-3",
        prompt=positive_prompt,
        size="1024x1024",
        quality="standard",
        response_format='b64_json',
        n=1,
    )

    image_base64 = response.data[0].b64_json
    image = Image.open(io.BytesIO(base64.b64decode(image_base64.split(",", 1)[0])))
    return image, image_base64


def remove_background_from_picture(image_base64: str) -> Tuple[Image, str]:
    # uses REMBG_ENDPOINT to remove the background from the image and returns it
    # edit DEFAULT_REMBG_PAYLOAD
    payload = DEFAULT_REMBG_PAYLOAD.copy()
    payload["input_image"] = image_base64
    print(f"Contacting REMBG api at {REMBG_ENDPOINT} with payload")
    response = requests.post(url=REMBG_ENDPOINT, json=payload)
    response_json = response.json()
    image_base64 = response_json["image"]
    image = Image.open(io.BytesIO(base64.b64decode(image_base64.split(",", 1)[0])))
    return image, image_base64


def img2img_image(image_base64: str) -> Tuple[Image, str]:
    # calls the img2img api with the image and returns it
    payload = DEFAULT_IMG2IMG_PAYLOAD.copy()
    payload["init_images"] = [image_base64]
    print(f"Contacting img2img api at {IMG2IMG_ENDPOINT} with payload")
    response = requests.post(url=IMG2IMG_ENDPOINT, json=payload)
    response_json = response.json()
    image_base64 = response_json["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(image_base64.split(",", 1)[0])))
    return image, image_base64


def outpaint_image(image: Image, positive_prompt: str, negative_prompt: str) -> Tuple[Image, str]:
    # Uses IMG2IMG_ENDPOINT to outpaint the image and returns it
    # first we'll "zoom out the image" by adding a transparent border around it
    # then resize it to 512x512
    # then we'll contact the API with that image, DPM++ 2M SDE Karras, and default settings with batch size 1

    # edit image
    image = image.resize((image.width + 20, image.height + 20))
    # # add 30 px transparent border and return in base64
    image = ImageOps.expand(image, border=30, fill=(0, 0, 0, 0))
    # in order to outpaint, we need a grayscale mask of the image border
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((0, 0, image.width, image.height), fill=255)
    # restrict to image border
    draw.rectangle((30, 30, image.width - 30, image.height - 30), fill=0)
    # DEBUG: save the mask
    mask.save("debug_mask.png")

    # stable diffusion API expects it in the format data:image/png;base64,yourbase64encodedpngstring
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_base64 = f"data:image/png;base64,{image_base64}"

    buffered = io.BytesIO()
    mask.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # DEBUG: save the image
    image.save("debug.png")

    # get outpainted image
    payload = DEFAULT_IMG2IMG_PAYLOAD.copy()
    payload["init_images"] = [image_base64]
    payload["mask"] = mask_base64
    payload["prompt"] = positive_prompt
    payload["negative_prompt"] = negative_prompt
    print(f"Contacting img2img API at {IMG2IMG_ENDPOINT} with payload")
    response = requests.post(url=IMG2IMG_ENDPOINT, json=payload)
    response_json = response.json()
    image_base64 = response_json["images"][0]
    image = Image.open(io.BytesIO(base64.b64decode(image_base64.split(",", 1)[0])))
    return image, image_base64


def assemble_final_image(image: Image) -> Image:
    # gets the result of generate_alchemy_picture and assembles it into the alchemizer asset
    # we have to keep in mind the input image is going to be 512x512, so we have to resize it to 256x256
    # the alchemiter asset is at images/alchemiter_pad.png and is 448x448
    # we'll place the image in the horizontal center of the alchemiter asset, with its bottom aligned with the bottom
    # of the alchemiter asset, with 15 pixels of bottom padding
    # we'll then save the result to output/alchemiter_result.png

    # resize image to 256x256
    image = image.resize((256, 256))

    # trim the image by removing the outer transparent edges
    bbox = image.getbbox()
    trimmed_image = image.crop(bbox)

    # open alchemiter asset
    alchemiter_asset = ALCHEMITER_ASSET

    # calculate the position to paste the trimmed image for centering
    alchemiter_width, alchemiter_height = alchemiter_asset.size
    trimmed_width, trimmed_height = trimmed_image.size
    paste_x = (alchemiter_width - trimmed_width) // 2
    paste_y = (alchemiter_height - trimmed_height) // 2

    # create a new blank image with alpha channel
    result = Image.new("RGBA", alchemiter_asset.size, (0, 0, 0, 0))

    # paste the alchemiter asset on the result image
    result.paste(alchemiter_asset, (0, 0))

    # paste the trimmed image on the result image at the calculated position
    result.paste(trimmed_image, (paste_x, paste_y + 20), trimmed_image)

    return result


async def generate_alchemy_picture(client, result_item: str) -> Image:
    # add suffixes
    positive_prompt = result_item.lower() + POSITIVE_PROMPT_SUFFIX
    negative_prompt = NEGATIVE_PROMPT_SUFFIX

    prompt = positive_prompt if negative_prompt == "" else positive_prompt + ", " + negative_prompt

    # generate image
    print(f"Generating image with prompt: {prompt}")
    # image, image_base64 = generate_image(positive_prompt, negative_prompt)
    image, image_base64 = await generate_image_dalle3(client, positive_prompt, negative_prompt)
    image.save("output/1_alchemized_item_initial.png")

    # (OUTDATED) outpaint to avoid cropping
    # image_uncropped, image_uncropped_base64 = outpaint_image(image, positive_prompt, negative_prompt)
    # image_uncropped.save("output/2_alchemized_item_uncropped.png")

    # (OUTDATED) img2img to a better random style
    # altered_image, altered_image_base64 = img2img_image(image_base64)
    # altered_image.save("output/2_alchemized_item_altered.png")

    # remove background
    image_no_bg, image_no_bg_base_64 = remove_background_from_picture(image_base64)
    image_no_bg.save("output/2_alchemized_item_no_bg.png")

    # check if removing background led to a too transparent image, if more than 90% has reduced alpha, then we'll
    # just use the image without removing the background
    alpha = image_no_bg.getchannel("A")
    alpha = alpha.histogram()
    alpha = alpha[0] / sum(alpha)
    min_alpha_threshold = 0.25
    max_alpha_threshold = 0.75
    print(f"Image: {result_item}")
    print(f"Alpha: {alpha}")
    if alpha < min_alpha_threshold or alpha > max_alpha_threshold:
        image_no_bg = image

    # Assemble final image
    final_image = assemble_final_image(image_no_bg)
    final_image.save("output/3_alchemized_item_final.png")

    return final_image
