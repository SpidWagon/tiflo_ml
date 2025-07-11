from utils.speech_modules import TTSEngine, voice_generation

model = TTSEngine()

comments = ['Девочка идёт по берегу реки.', 'Девочка идёт по берегу реки.', 'Она останавливается и смотрит на воду.', 'Она останавливается и смотрит на воду.', 'Лёгкий ветер треплет её волосы.', 'Лёгкий ветер треплет её волосы.']

voice_generation(comments, model, "output.mp3")