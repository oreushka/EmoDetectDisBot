def recommend_games(emotion):
    # Здесь можно добавить код для рекомендации игр на основе эмоции
    recommendations = {
        "happy": [
            "Putt Party",
            " Poker Night",
            " Sketch Heads",
            " Chess in the Park",
            " Land-io",
            " Blazing 8s",
            " Letter League",
            " Checkers in the Park",
            " Spellcast",
            " Bobble League",
            " Know What I Meme",
            " Gartic Phone",
            " Color Together",
            " Krunker Strike FRVR",
            " Colonist",
            " Bobble Bash",
            " Chef Showdown",
        ],
        "fear": ["Watch Together", " Jamspace Whiteboard"],
        "angry": [
            "Watch Together",
            " Sketch Heads",
            " Land-io",
            " Bobble League",
            " Gartic Phone",
            " Jamspace Whiteboard",
            " Krunker Strike FRVR",
            " Bobble Bash",
            " Chef Showdown",
        ],
        "neutral": ["Выбери любую игру, тебе понравится!"],
        "disgust": ["Watch Together", " Jamspace Whiteboard"],
    }
    return recommendations.get(emotion, [])
