import unicodedata

with open("resources/lexicons/ha.txt", encoding="utf-8") as f:
    lexicon = {unicodedata.normalize("NFC", line.strip().lower()) for line in f if line.strip()}

token = unicodedata.normalize("NFC", "gida".strip(".,!?;:\"()").lower())
print(repr(token))
print(token in lexicon)
