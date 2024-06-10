import discord
from discord.ext import commands
from model import EmotionModel, analyze_emotions
from preprocess import preprocess_ppg, preprocess_text
from utils import recommend_games

# Инициализация бота
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# Загрузка модели
model = EmotionModel()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.command(name="emotion")
async def emotion(ctx, method: str, *, input_data: str):
    if method == "text":
        processed_text = preprocess_text(input_data)
        emotion = model.predict_text(processed_text)
        await ctx.send(f"Detected emotion: {emotion}")
    elif method == "ppg":
        processed_ppg = preprocess_ppg(input_data)
        emotion = analyze_emotions(processed_ppg)
        await ctx.send(f"Detected emotion: {emotion}")
    else:
        await ctx.send('Invalid method. Use "text" or "ppg".')


@bot.command(name="recommend_games")
async def recommend(ctx, emotion: str):
    games = recommend_games(emotion)
    await ctx.send(f"Recommended games: {games}")


bot.run("YOUR_DISCORD_BOT_TOKEN")
