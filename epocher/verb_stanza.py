import stanza

# Initialize the Stanza pipeline for English
nlp = stanza.Pipeline('en')

# Example sentence
sentence = "Taking a measured sip of white wine from his crystal glass, he glanced at his host."

# Process the sentence
doc = nlp(sentence)

# Try to extract verb features which are interpretable.

# Loop through sentences and words to find verbs and their morphological features
for sent in doc.sentences:
    for word in sent.words:
        if word.upos == 'VERB':  # Check if the word is a verb

            print(f"Verb: {word.text}")

            # Split features into individual components
            features = {feat.split('=')[0]: feat.split('=')[1] for feat in word.feats.split('|') if '=' in feat}
            # Extract features or use 'N/A' if they don't exist
            print(f" - Tense: {features.get('Tense', 'N/A')}")
            print(f" - Aspect: {features.get('Aspect', 'N/A')}")
            print(f" - Voice: {features.get('Voice', 'N/A')}")
            print(f" - Mood: {features.get('Mood', 'N/A')}")
            print(f" - Person: {features.get('Person', 'N/A')}")
            print(f" - Number: {features.get('Number', 'N/A')}")
            print(f" - Other Features: {word.feats}")
