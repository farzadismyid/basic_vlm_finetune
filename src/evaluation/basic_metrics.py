def response_length(text):

    return len(text.split())


def contains_fashion_keywords(text):

    keywords = [
        "jacket",
        "shirt",
        "dress",
        "pants",
        "style",
        "fashion",
        "outfit",
        "casual",
        "formal",
    ]

    text = text.lower()

    matches = [
        word
        for word in keywords
        if word in text
    ]

    return matches
