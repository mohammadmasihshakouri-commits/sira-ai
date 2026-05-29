import random


GREETING_VARIANTS = [
    "سلام، وقتتون بخیر.",

    "سلام، در خدمتم 🌸",

    "سلام، وقتتون بخیر 😊",

    "سلام 🌸",
]


THANKS_VARIANTS = [
    "خواهش می‌کنم، وظیفه بود.",

    "خواهش می‌کنم 🌸",

    "انجام وظیفه‌ست 😊",
]


ASK_MORE_HELP_VARIANTS = [
    "مورد دیگه‌ای هست که بتونم راهنمایی‌تون کنم؟",

    "اگر سوال دیگه‌ای دارید در خدمتم.",

    "کمک دیگه‌ای از دستم برمیاد؟",

    "اگر مورد دیگه‌ای هست بفرمایید 😊",
]


def random_greeting():
    return random.choice(GREETING_VARIANTS)


def random_thanks_reply():
    return random.choice(THANKS_VARIANTS)


def random_more_help():
    return random.choice(ASK_MORE_HELP_VARIANTS)