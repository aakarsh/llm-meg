import spacy

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Example sentence
sentence = "Taking a measured sip of white wine from his crystal glass, he glanced at his host."

# Process the sentence
doc = nlp(sentence)

# Loop through tokens to find verbs and their features
for token in doc:
    if token.pos_ == "VERB":
        print(f"Verb: {token.text}")
        print(f" - Tense: {token.morph.get('Tense')}")
        print(f" - Aspect: {token.morph.get('Aspect')}")
        print(f" - Voice: {token.morph.get('Voice')}")
        print(f" - Mood: {token.morph.get('Mood')}")
        print(f" - Person: {token.morph.get('Person')}")
        print(f" - Number: {token.morph.get('Number')}")
        print(f" - Other Features: {token.morph}")
