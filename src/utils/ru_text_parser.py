from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()

def lemmatize_text(text: str) -> str:
    tokens = []
    for word in text.split():
        parsed = morph.parse(word)
        if parsed:
            tokens.append(parsed[0].normal_form)
        else:
            tokens.append(word)
    return " ".join(tokens)
