import httpx
from settings import settings

# Optional: pick a specific voice ID or use the default
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # You can explore voices at https://elevenlabs.io/voice-library
JOSH_VOICE_ID = "TxGEqnHWrfWFTfGW9XjX"

async def text_to_effect(effect_description: str, duration_seconds: int = 5, prompt_influence: float = 0.8) -> bytes:
    """
    Generates a sound effect using ElevenLabs' sound generation API,
    based on a rich textual description.

    Args:
        effect_description (str): A vivid description of the desired sound effect.
        duration_seconds (int): Duration of the generated sound in seconds (max 22).
        prompt_influence (float): Value between 0 and 1; higher means closer to prompt.

    Returns:
        bytes: The MP3 audio binary data.
    """
    url = "https://api.elevenlabs.io/v1/sound-generation"

    headers = {
        "xi-api-key": settings.ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "text": effect_description,
        "duration_seconds": duration_seconds,
        "prompt_influence": prompt_influence
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to generate sound effect: {response.status_code}, {response.text}")


async def text_to_speech(text: str, voice_id: str = JOSH_VOICE_ID) -> bytes:
    """
    Generates a fantastic sound effect for an ad using ElevenLabs text-to-speech API.
    
    Args:
        ad_description (str): A detailed description of the desired sound effect.
    
    Returns:
        bytes: The MP3 audio data.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": settings.ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }

    payload = {
        "text": text,
        "model_id": "eleven_flash_v2_5",  # High-quality multilingual model
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
            "style": 1.0,  # More expressive
            "use_speaker_boost": True
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.content  # MP3 binary
    else:
        raise Exception(f"Failed to generate audio: {response.status_code}, {response.text}")

