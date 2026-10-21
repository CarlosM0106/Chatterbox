import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando el modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto="Hola, bienvenidos a Inteligencia Artificial generativa, vamos a crear imágenes, texto, audio y si el computador aguanta video equis d"

print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, # Expresividad o emoción
    cfg_weight=0.4, # Afecta velocidad y ritmo
    temperature=0.8, # Controla variabilidad
    repetition_penalty=1.2, # Penaliza repeticiones extrañas
    min_p=0.05, # Filtra opciones improbables durante la generación
    top_p=1.0, # Controla el muestreo probabilistico
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio generado")
