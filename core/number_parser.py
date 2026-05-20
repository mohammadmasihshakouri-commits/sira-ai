import re

from core.normalization import normalize_text_light

# -----------------------------
# Number extraction
# -----------------------------
def extract_iran_mobile_grouped_number(text):
    """
    Handles Iranian mobile numbers spoken in grouped Persian style.

    Examples:
    صفر نهصد و دوازده، صد و دو، چهل و دو، هشتاد -> 09121024280
    نهصد و دوازده، صد و دو، چهل و دو، هشتاد -> 9121024280
    """

    text = normalize_text_light(text)
    text = text.replace("صرف", "صفر").replace("سفر", "صفر")

    ones = {
        "صفر": 0,
        "یک": 1,
        "یه": 1,
        "دو": 2,
        "سه": 3,
        "چهار": 4,
        "چار": 4,
        "پنج": 5,
        "شش": 6,
        "شیش": 6,
        "هفت": 7,
        "هشت": 8,
        "نه": 9,
    }

    teens = {
        "ده": 10,
        "یازده": 11,
        "دوازده": 12,
        "سیزده": 13,
        "چهارده": 14,
        "پانزده": 15,
        "پونزده": 15,
        "شانزده": 16,
        "هفده": 17,
        "هجده": 18,
        "نوزده": 19,
    }

    tens = {
        "بیست": 20,
        "سی": 30,
        "چهل": 40,
        "پنجاه": 50,
        "پنجا": 50,
        "شصت": 60,
        "هفتاد": 70,
        "هشتاد": 80,
        "نود": 90,
    }

    hundreds = {
        "صد": 100,
        "یکصد": 100,
        "دویست": 200,
        "سیصد": 300,
        "چهارصد": 400,
        "پانصد": 500,
        "ششصد": 600,
        "هفتصد": 700,
        "هشتصد": 800,
        "نهصد": 900,
    }

    valid_words = set(ones) | set(teens) | set(tens) | set(hundreds) | {"و"}

    tokens = re.findall(r"[\w\u0600-\u06FF]+", text)
    tokens = [token for token in tokens if token in valid_words]

    if not tokens:
        return ""

    # Mobile-style spoken numbers usually start with صفر or نهصد.
    # صفر نهصد و دوازده... -> 0912...
    # نهصد و دوازده... -> 912...
    if tokens[0] not in ["صفر", "نهصد"]:
        return ""

    if not any(token in hundreds for token in tokens):
        return ""

    def parse_hundreds_group(start_index):
        value = hundreds[tokens[start_index]]
        i = start_index + 1

        if i < len(tokens) and tokens[i] == "و":
            i += 1

        if i < len(tokens):
            token = tokens[i]

            if token in teens:
                value += teens[token]
                i += 1

            elif token in tens:
                value += tens[token]
                i += 1

                if i < len(tokens) and tokens[i] == "و":
                    i += 1

                if i < len(tokens) and tokens[i] in ones:
                    value += ones[tokens[i]]
                    i += 1

            elif token in ones:
                value += ones[token]
                i += 1

        return str(value).zfill(3), i

    def parse_two_digit_group(start_index):
        token = tokens[start_index]
        i = start_index

        if token in tens:
            value = tens[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens) and tokens[i] in ones:
                value += ones[tokens[i]]
                i += 1

            return str(value).zfill(2), i

        if token in teens:
            return str(teens[token]).zfill(2), i + 1

        if token in ones:
            return str(ones[token]), i + 1

        return "", i + 1

    result = ""
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "و":
            i += 1
            continue

        if token == "صفر":
            result += "0"
            i += 1
            continue

        if token in hundreds:
            group, i = parse_hundreds_group(i)
            result += group
            continue

        if token in tens or token in teens or token in ones:
            group, i = parse_two_digit_group(i)
            result += group
            continue

        i += 1

    if 10 <= len(result) <= 11:
        return result

    return ""

def parse_reservation_code_spoken_persian(text):
    """
    Parses 8-digit reservation codes spoken in Persian.
    Supports grouped two-digit and digit-by-digit formats.
    """

    text = normalize_text_light(text)

    ascii_text = text.translate(
        str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    )

    existing_digits = "".join(re.findall(r"\d", ascii_text))
    if len(existing_digits) == 8:
        return existing_digits

    ones = {
        "صفر": 0,
        "یک": 1,
        "یه": 1,
        "دو": 2,
        "سه": 3,
        "چهار": 4,
        "چار": 4,
        "پنج": 5,
        "شش": 6,
        "شیش": 6,
        "هفت": 7,
        "هشت": 8,
        "نه": 9,
    }

    teens = {
        "ده": 10,
        "یازده": 11,
        "دوازده": 12,
        "سیزده": 13,
        "چهارده": 14,
        "پانزده": 15,
        "پونزده": 15,
        "شانزده": 16,
        "هفده": 17,
        "هجده": 18,
        "نوزده": 19,
    }

    tens = {
        "بیست": 20,
        "سی": 30,
        "چهل": 40,
        "پنجاه": 50,
        "پنجا": 50,
        "شصت": 60,
        "هفتاد": 70,
        "هشتاد": 80,
        "نود": 90,
    }

    valid_words = set(ones) | set(teens) | set(tens) | {"و"}

    def parse_two_digit_chunk(chunk):
        tokens = re.findall(r"[\u0600-\u06FF]+", chunk)
        tokens = [token for token in tokens if token in valid_words]
        tokens = [token for token in tokens if token != "و"]

        if not tokens:
            return None

        if len(tokens) == 1:
            token = tokens[0]

            if token in ones:
                return ones[token]

            if token in teens:
                return teens[token]

            if token in tens:
                return tens[token]

        if len(tokens) == 2:
            first, second = tokens

            if first in tens and second in ones:
                return tens[first] + ones[second]

            if first in ones and second in ones:
                return (ones[first] * 10) + ones[second]

        return None

    # Strategy 1: grouped two-digit chunks
    chunks = [
        chunk.strip()
        for chunk in re.split(r"[،,.]+", text)
        if chunk.strip()
    ]

    if len(chunks) >= 4:
        parts = []

        for chunk in chunks:
            value = parse_two_digit_chunk(chunk)

            if value is not None:
                parts.append(str(value).zfill(2))

        candidate = "".join(parts)

        if len(candidate) == 8:
            return candidate

    # Strategy 2: sequential two-digit groups without clear punctuation
    tokens = re.findall(r"[\u0600-\u06FF]+", text)
    tokens = [token for token in tokens if token in valid_words]

    sequential_parts = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "و":
            i += 1
            continue

        value = None

        # بیست و دو -> 22 / شصت و هشت -> 68
        if token in tens:
            value = tens[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens) and tokens[i] in ones:
                value += ones[tokens[i]]
                i += 1

            sequential_parts.append(str(value).zfill(2))
            continue

        # دوازده -> 12
        if token in teens:
            sequential_parts.append(str(teens[token]).zfill(2))
            i += 1
            continue

        # پنج و چهار -> 54
        if token in ones:
            first = ones[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens) and tokens[i] in ones:
                second = ones[tokens[i]]
                i += 1
                sequential_parts.append(f"{first}{second}")
                continue

            sequential_parts.append(str(first))
            continue

        i += 1

    sequential_candidate = "".join(sequential_parts)

    if len(sequential_candidate) == 8:
        return sequential_candidate

    # Strategy 3: digit-by-digit Persian words
    digit_tokens = [
        token for token in tokens
        if token in ones
    ]

    if len(digit_tokens) == 8:
        return "".join(str(ones[token]) for token in digit_tokens)

    return ""

def extract_digits_from_spoken_persian_number(text):
    """
    Converts common spoken Persian number patterns into digits.
    Handles:
    - 0902 201 42 80 -> 09022014280
    - صفر نهصد و دوازده صد و سه چهل و دو هشتاد -> 09121034280
    - سی و سه پنجاه و دو چهل و سه چهل و سه -> 33524343
    """

    text = normalize_text_light(text)

    mobile_grouped_number = extract_iran_mobile_grouped_number(text)

    if mobile_grouped_number:
        return mobile_grouped_number

    ascii_text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))

    all_digits = "".join(re.findall(r"\d", ascii_text))
    if 4 <= len(all_digits) <= 20:
        return all_digits

    ones = {
        "صفر": 0,
        "یک": 1,
        "یه": 1,
        "دو": 2,
        "سه": 3,
        "چهار": 4,
        "چار": 4,
        "پنج": 5,
        "شش": 6,
        "شیش": 6,
        "هفت": 7,
        "هشت": 8,
        "نه": 9,
    }

    teens = {
        "ده": 10,
        "یازده": 11,
        "دوازده": 12,
        "سیزده": 13,
        "چهارده": 14,
        "پانزده": 15,
        "پونزده": 15,
        "شانزده": 16,
        "هفده": 17,
        "هجده": 18,
        "نوزده": 19,
    }

    tens = {
        "بیست": 20,
        "سی": 30,
        "چهل": 40,
        "پنجاه": 50,
        "پنجا": 50,
        "شصت": 60,
        "هفتاد": 70,
        "هشتاد": 80,
        "نود": 90,
    }

    hundreds = {
        "صد": 100,
        "یکصد": 100,
        "دویست": 200,
        "سیصد": 300,
        "چهارصد": 400,
        "پانصد": 500,
        "ششصد": 600,
        "هفتصد": 700,
        "هشتصد": 800,
        "نهصد": 900,
    }

    valid_words = set(ones) | set(teens) | set(tens) | set(hundreds) | {"و"}

    tokens = re.findall(r"[\w\u0600-\u06FF]+", text)
    tokens = [token for token in tokens if token in valid_words]

    if not tokens:
        return ""

    # Mobile number style:
    # صفر نهصد و دوازده صد و سه چهل و دو هشتاد
    # -> 0 + 912 + 103 + 42 + 80
    # -> 09121034280
    if tokens[0] == "صفر" and any(token in hundreds for token in tokens):
        result = ""
        i = 0

        while i < len(tokens):
            token = tokens[i]

            if token == "و":
                i += 1
                continue

            if token == "صفر":
                result += "0"
                i += 1
                continue

            if token in hundreds:
                value = hundreds[token]
                i += 1

                if i < len(tokens) and tokens[i] == "و":
                    i += 1

                if i < len(tokens):
                    next_token = tokens[i]

                    if next_token in teens:
                        value += teens[next_token]
                        i += 1

                    elif next_token in tens:
                        value += tens[next_token]
                        i += 1

                        if i < len(tokens) and tokens[i] == "و":
                            i += 1

                        if i < len(tokens) and tokens[i] in ones:
                            value += ones[tokens[i]]
                            i += 1

                    elif next_token in ones:
                        value += ones[next_token]
                        i += 1

                result += str(value).zfill(3)
                continue

            if token in tens:
                value = tens[token]
                i += 1

                if i < len(tokens) and tokens[i] == "و":
                    i += 1

                if i < len(tokens) and tokens[i] in ones:
                    value += ones[tokens[i]]
                    i += 1

                result += str(value).zfill(2)
                continue

            if token in teens:
                result += str(teens[token]).zfill(2)
                i += 1
                continue

            if token in ones:
                result += str(ones[token])
                i += 1
                continue

            i += 1

        if 10 <= len(result) <= 11:
            return result

    # Digit-by-digit style:
    # صفر نه یک دو سه چهار...
    digit_style_tokens = [token for token in tokens if token != "و"]

    if digit_style_tokens and all(token in ones for token in digit_style_tokens):
        digits = "".join(str(ones[token]) for token in digit_style_tokens)

        if 4 <= len(digits) <= 20:
            return digits

    # Reservation-code style in 2-digit groups:
    # سی و سه پنجاه و دو چهل و سه چهل و سه -> 33524343
    if not any(token in hundreds for token in tokens):
        two_digit_groups = []
        j = 0

        while j < len(tokens):
            token = tokens[j]

            if token == "و":
                j += 1
                continue

            if token in tens:
                value = tens[token]
                j += 1

                if j < len(tokens) and tokens[j] == "و":
                    j += 1

                if j < len(tokens) and tokens[j] in ones:
                    value += ones[tokens[j]]
                    j += 1

                two_digit_groups.append(str(value).zfill(2))
                continue

            if token in teens:
                two_digit_groups.append(str(teens[token]).zfill(2))
                j += 1
                continue

            if token in ones:
                two_digit_groups.append(str(ones[token]))
                j += 1
                continue

            j += 1

        two_digit_result = "".join(two_digit_groups)

        if 4 <= len(two_digit_result) <= 20:
            return two_digit_result

    # General grouped parser fallback
    result = ""
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "و":
            i += 1
            continue

        if token == "صفر":
            result += "0"
            i += 1
            continue

        if token in hundreds:
            value = hundreds[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens):
                next_token = tokens[i]

                if next_token in teens:
                    value += teens[next_token]
                    i += 1
                elif next_token in tens:
                    value += tens[next_token]
                    i += 1

                    if i < len(tokens) and tokens[i] == "و":
                        i += 1

                    if i < len(tokens) and tokens[i] in ones:
                        value += ones[tokens[i]]
                        i += 1
                elif next_token in ones:
                    value += ones[next_token]
                    i += 1

            result += str(value).zfill(3)
            continue

        if token in tens:
            value = tens[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens) and tokens[i] in ones:
                value += ones[tokens[i]]
                i += 1

            result += str(value).zfill(2)
            continue

        if token in teens:
            result += str(teens[token]).zfill(2)
            i += 1
            continue

        if token in ones:
            result += str(ones[token])
            i += 1
            continue

        i += 1

    if 4 <= len(result) <= 20:
        return result

    return ""

def parse_persian_identifier_words(text):
    """
    Central parser for Persian spoken identifiers.

    Handles:
    - هشت هفت شش پنج چهار سه دو یک -> 87654321
    - سی و دو بیست و سه چهل و پنج شصت و هفت -> 32234567
    - پنج چهار دو سه پنج چهار بیست و سه -> 54235423
    - صفر نهصد و دوازده صد و دو چهل و دو هشتاد -> 09121024280
    """

    text = normalize_text_light(text)
    text = text.replace("صرف", "صفر").replace("سفر", "صفر")

    ascii_text = text.translate(
        str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    )

    existing_digits = "".join(re.findall(r"\d", ascii_text))

    if 4 <= len(existing_digits) <= 20:
        return existing_digits

    ones = {
        "صفر": "0",
        "یک": "1",
        "یه": "1",
        "دو": "2",
        "سه": "3",
        "چهار": "4",
        "چار": "4",
        "پنج": "5",
        "شش": "6",
        "شیش": "6",
        "هفت": "7",
        "هشت": "8",
        "نه": "9",
    }

    teens = {
        "ده": 10,
        "یازده": 11,
        "دوازده": 12,
        "سیزده": 13,
        "چهارده": 14,
        "پانزده": 15,
        "پونزده": 15,
        "شانزده": 16,
        "هفده": 17,
        "هجده": 18,
        "نوزده": 19,
    }

    tens = {
        "بیست": 20,
        "سی": 30,
        "چهل": 40,
        "پنجاه": 50,
        "پنجا": 50,
        "شصت": 60,
        "هفتاد": 70,
        "هشتاد": 80,
        "نود": 90,
    }

    hundreds = {
        "صد": 100,
        "یکصد": 100,
        "دویست": 200,
        "سیصد": 300,
        "چهارصد": 400,
        "پانصد": 500,
        "ششصد": 600,
        "هفتصد": 700,
        "هشتصد": 800,
        "نهصد": 900,
    }

    valid_words = set(ones) | set(teens) | set(tens) | set(hundreds) | {"و"}

    tokens = re.findall(r"[\u0600-\u06FF]+", text)
    tokens = [token for token in tokens if token in valid_words]

    if not tokens:
        return ""

    # 1) Iranian mobile grouped style:
    # صفر نهصد و دوازده صد و دو چهل و دو هشتاد -> 09121024280
    mobile_number = extract_iran_mobile_grouped_number(text)
    if mobile_number:
        return mobile_number

    pieces = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token == "و":
            i += 1
            continue

        # Digit-by-digit: پنج، چهار، دو...
        if token in ones:
            pieces.append(ones[token])
            i += 1
            continue

        # Two-digit group: بیست و سه -> 23
        if token in tens:
            value = tens[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens) and tokens[i] in ones:
                value += int(ones[tokens[i]])
                i += 1

            pieces.append(str(value).zfill(2))
            continue

        # Teen group: دوازده -> 12
        if token in teens:
            pieces.append(str(teens[token]).zfill(2))
            i += 1
            continue

        # Hundreds group fallback: نهصد و دوازده -> 912
        if token in hundreds:
            value = hundreds[token]
            i += 1

            if i < len(tokens) and tokens[i] == "و":
                i += 1

            if i < len(tokens):
                next_token = tokens[i]

                if next_token in teens:
                    value += teens[next_token]
                    i += 1

                elif next_token in tens:
                    value += tens[next_token]
                    i += 1

                    if i < len(tokens) and tokens[i] == "و":
                        i += 1

                    if i < len(tokens) and tokens[i] in ones:
                        value += int(ones[tokens[i]])
                        i += 1

                elif next_token in ones:
                    value += int(ones[next_token])
                    i += 1

            pieces.append(str(value).zfill(3))
            continue

        i += 1

    candidate = "".join(pieces)

    if 4 <= len(candidate) <= 20:
        return candidate

    return ""
def extract_identifier_candidate(text):
    """
    Handles mixed spoken/digit identifiers.
    Examples:
    صفر 902 202 4280 -> 09022024280
    912 201 4282 -> 9122014282
    صفر نهصد و دو دویست و یک چهل و دو هشتاد -> 09022014280
    """

    text = normalize_text_light(text)

    reservation_code_candidate = parse_reservation_code_spoken_persian(text)

    if reservation_code_candidate:
        return reservation_code_candidate
    
    # Extra safety for common STT mistakes around صفر.
    correction_replacements = {
        "صرف نهصد": "صفر نهصد",
        "سفر نهصد": "صفر نهصد",
        "صرف نصد": "صفر نهصد",
        "سفر نصد": "صفر نهصد",
        "صرف": "صفر",
        "سفر": "صفر",
    }

    for wrong, correct in correction_replacements.items():
        text = text.replace(wrong, correct)
    central_candidate = parse_persian_identifier_words(text)

    if central_candidate:
        return central_candidate
    ascii_text = text.translate(
        str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    )
    # Digit-by-digit Persian identifiers:
    # هشت، هفت، شش، پنج، چهار، سه، دو، یک -> 87654321
    # Useful for 8-digit reservation codes.
    digit_word_map = {
        "صفر": "0",
        "یک": "1",
        "یه": "1",
        "دو": "2",
        "سه": "3",
        "چهار": "4",
        "چار": "4",
        "پنج": "5",
        "شش": "6",
        "شیش": "6",
        "هفت": "7",
        "هشت": "8",
        "نه": "9",
    }

    digit_tokens = re.findall(r"\d+|[\u0600-\u06FF]+", ascii_text)

    digit_pieces = []
    digit_like_count = 0

    for index, token in enumerate(digit_tokens):
        if token.isdigit():
            digit_pieces.append(token)
            digit_like_count += 1
            continue

        if token in digit_word_map:
            digit_pieces.append(digit_word_map[token])
            digit_like_count += 1
            continue

        # STT sometimes hears "چهار" as "چهل" in digit-by-digit codes.
        # Only treat standalone "چهل" as 4 when it is NOT part of "چهل و دو".
        if token == "چهل":
            next_token = digit_tokens[index + 1] if index + 1 < len(digit_tokens) else ""

            if next_token != "و":
                digit_pieces.append("4")
                digit_like_count += 1

    digit_by_digit_candidate = "".join(digit_pieces)

    # If most tokens are digit-like and length is valid, trust this path.
    if digit_like_count >= 4 and 4 <= len(digit_by_digit_candidate) <= 20:
        return digit_by_digit_candidate
    
    # If the sentence contains grouped Persian number words, let the spoken-number
    # parser handle it. This is important for phrases like:
    # صفر نهصد و دوازده، صد و دو، چهل و دو، هشتاد
    grouped_number_words = [
        "صفر",
        "صد",
        "یکصد",
        "نهصد",
        "دویست",
        "سیصد",
        "چهارصد",
        "پانصد",
        "ششصد",
        "هفتصد",
        "هشتصد",
        "بیست",
        "سی",
        "چهل",
        "پنجاه",
        "پنجا",
        "شصت",
        "هفتاد",
        "هشتاد",
        "نود",
        "ده",
        "یازده",
        "دوازده",
        "سیزده",
        "چهارده",
        "پانزده",
        "پونزده",
        "شانزده",
        "هفده",
        "هجده",
        "نوزده",
    ]

    if any(word in text for word in grouped_number_words):
        print("DEBUG grouped text:", text)
        print("DEBUG spoken digits:", extract_digits_from_spoken_persian_number(text))
        spoken_digits = extract_digits_from_spoken_persian_number(text)

        if spoken_digits:
            return spoken_digits

    digit_words = {
        "صفر": "0",
        "یک": "1",
        "یه": "1",
        "دو": "2",
        "سه": "3",
        "چهار": "4",
        "چار": "4",
        "پنج": "5",
        "شش": "6",
        "شیش": "6",
        "هفت": "7",
        "هشت": "8",
        "نه": "9",
    }

    tokens = re.findall(r"\d+|[\u0600-\u06FF]+", ascii_text)

    pieces = []

    for token in tokens:
        if token.isdigit():
            pieces.append(token)
        elif token in digit_words:
            pieces.append(digit_words[token])

    mixed_candidate = "".join(pieces)

    if 4 <= len(mixed_candidate) <= 20:
        return mixed_candidate

    return extract_digits_from_spoken_persian_number(text)

def repair_possible_8_digit_reservation_code(digits):
    """
    Repairs common STT hallucinations for 8-digit reservation codes.
    Example:
    221364823 -> 22136483
    """

    clean_digits = re.sub(r"\D", "", digits)

    if len(clean_digits) != 9:
        return ""

    # Common hallucination:
    # 83 -> 823
    if clean_digits.endswith("823"):
        return clean_digits[:-3] + "83"

    # 73 -> 723
    if clean_digits.endswith("723"):
        return clean_digits[:-3] + "73"

    # 63 -> 623
    if clean_digits.endswith("623"):
        return clean_digits[:-3] + "63"

    return ""

def normalize_requested_identifier(digits):
    clean_digits = re.sub(r"\D", "", digits)

    if len(clean_digits) == 10 and clean_digits.startswith("9"):
        return "0" + clean_digits, "phone"

    if len(clean_digits) == 11:
        return clean_digits, "phone"

    if len(clean_digits) == 8:
        return clean_digits, "reservation_code"

    repaired_code = repair_possible_8_digit_reservation_code(clean_digits)

    if repaired_code:
        return repaired_code, "reservation_code"

    return clean_digits, "invalid_identifier"

def enrich_text_with_detected_number(text):
    """
    Adds detected digit form to the text so the rest of the system can use it.
    """

    digits = extract_identifier_candidate(text)

    if digits:
        return f"{text} شماره تشخیص داده شده: {digits}", digits

    return text, ""

