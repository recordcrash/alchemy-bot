# AlchemyBot: GPT-4 + Stable Diffusion powered Homestuck Alchemy Discord chatbot

![](https://raw.githubusercontent.com/recordcrash/alchemy-bot/master/metadata/alchemy.png)
![](https://raw.githubusercontent.com/recordcrash/alchemy-bot/master/metadata/coat.png)

## What is this?

A Discord bot that allows users to generate text + image of combinations of two items like in the webcomic [Homestuck](https://bambosh.github.io/unofficial-homestuck-collection/), using LLM and diffusion AI technologies.

## How to use

1. Create a virtual environment and install the requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create a Discord bot and get its token. See [this guide](https://discordpy.readthedocs.io/en/stable/discord.html) for more information.
3. Create a `.env` file from `.env.example` and fill in the Discord and OpenAI tokens.
4. If you want image generation, download AUTOMATIC1111's Stable Diffusion UI and keep it up at the default port. You might need to install the RemBG extension, and the models linked in the blog post. If you don't want image generation, you might need to lightly edit the code in `alchemy_bot.py`.
5. Run the bot:

```bash
python3 bot.py
```

6. Invite the bot to your server and launch it with `D--> alchemize ITEM1 && ITEM2`. `alchemize3` uses GPT-3.5 and `alchemize4` uses GPT-4.

Learn more in the [blog post](https://recordcrash.substack.com/homestuck-alchemy-stable-diffusion-gpt-discord).

## Credits

- Clue, Tensei and the [Homestuck Discord](https://discord.gg/homestuck) for varied help.
- The good parts of [Homestuck](https://www.homestuck.com/) for the inspiration.
- [AUTOMATIC1111](https://twitter.com/AUTOMATIC1111) for his amazing Stable Diffusion UI.
- [OpenAI](https://openai.com/) and their LLM team.
- The creators of [Fantassified Icons](https://civitai.com/models/4713/fantassified-icons) for their surprisingly good icon model.