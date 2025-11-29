import random
import urllib.parse

import httpx
import structlog

logger = structlog.get_logger(__name__)


async def get_available_avatar_styles() -> list[str]:
    """
    Get list of available DiceBear avatar styles from API
    with fallback of hardcoded list.
    :return: List of available styles.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.dicebear.com/7.x/styles", timeout=5.0)

            if response.status_code == 200:
                styles_data = response.json()
                return [style["id"] for style in styles_data if "id" in style]

    except (httpx.RequestError, KeyError, ValueError) as e:
        logger.warning(f"Could not fetch DiceBear styles from API: {e}")
        logger.info("Fallback to hardcoded style list")

    return [
        "bottts",
        "avataaars",
        "big-smile",
        "identicon",
        "initials",
        "pixel-art",
        "adventurer",
        "big-ears",
        "croodles",
        "fun-emoji",
        "lorelei",
        "micah",
        "miniavs",
        "open-peeps",
        "personas",
        "rings",
        "shapes",
    ]


async def get_random_avatar_style() -> str:
    """
    Get a random avatar style from available styles.
    :return: Random style name
    """
    styles = await get_available_avatar_styles()
    return random.choice(styles)


async def is_valid_avatar_style(style: str) -> bool:
    """
    Check if style is valid/available.
    :param style: Style to validate
    :return: True if available, else False
    """
    available_styles = await get_available_avatar_styles()
    return style.lower() in [available_style.lower() for available_style in available_styles]


async def generate_avatar_url(username: str, style: str = "bottts") -> str:
    """
    Generate DiceBear avatar URL based on username.
    :param username: Username used for avatar
    :param style: DiceBear style (default: Bottts)
    :return: Avatar URL
    """
    if not await is_valid_avatar_style(style):
        print(f"Warning: Invalid style '{style}', falling back to 'bottts'")
        style = "bottts"

    safe_username = urllib.parse.quote_plus(username.lower())

    avatar_url = f"https://api.dicebear.com/7.x/{style}/svg?seed={safe_username}"

    return avatar_url
