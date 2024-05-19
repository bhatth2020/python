# pylint: skip-file
# users the transformer python library and some of its capabilities
# transformers library is noted here - https://github.com/huggingface/transformers
# !pip install transformers => to install the library
# The transformers Python package is an open-source library developed by Hugging Face, a company specializing in natural language processing (NLP) and machine learning. 
# The library provides easy access to a variety of pre-trained models for tasks such as text classification, named entity recognition, text generation, and translation

# example-1
from transformers import pipeline

# Load the pre-trained BERT model and tokenizer for text classification
classifier = pipeline("sentiment-analysis")

# Define a list of texts to classify
texts = [
    "I love using transformers for natural language processing!",
    "This library is incredibly useful and versatile.",
    "I'm not sure if I like this package.",
    "This is the worst experience I've had with any library."
]

# Classify the texts
results = classifier(texts)

# Print the results
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Confidence: {result['score']:.4f}\n")

#example-2
from transformers import pipeline

# Sentiment Analysis with a specified model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print(sentiment_pipeline("I don't like using transformers for natural language processing"))

# Text Generation with a specified model
text_generator = pipeline("text-generation", model="gpt2")
print(text_generator("Cat and dog", max_length=50, num_return_sequences=2))

# Question Answering with a specified model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
context = "Google was founded by Larry Page and Sergey Brin in 1998 while they were Ph.D. students at Stanford University. The initial concept for Google originated from their research project called BackRub, which was a search engine that analyzed the web's backlink structure to rank the importance of websites. The project was later renamed Google, inspired by the mathematical term googol which refers to the number 1 followed by 100 zeros, reflecting their mission to organize vast amounts of information on the web"
question = "Who funded Google"
print(qa_pipeline(question=question, context=context))

# Text Summarization with a specified model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = text = """Google was founded by Larry Page and Sergey Brin in 1998 while they were Ph.D. students at Stanford University. The initial concept for Google originated from their research project called BackRub, which was a search engine that analyzed the web's backlink structure to rank the importance of websites. The project was later renamed Google, inspired by the mathematical term "googol," which refers to the number 1 followed by 100 zeros, reflecting their mission to organize vast amounts of information on the web.
Incorporated on September 4, 1998, Google received its first significant funding from Andy Bechtolsheim, co-founder of Sun Microsystems, who wrote a $100,000 check to Google Inc. before the company was officially formed. This investment enabled them to move out of their dorm rooms and set up their first office in a garage in Menlo Park, California, owned by Susan Wojcicki, who later became an executive at Google and CEO of YouTube.
From these humble beginnings, Google rapidly grew, moving to larger offices in Palo Alto and eventually to its current headquarters, known as the Googleplex, in Mountain View, California. The company's innovative approach, including its use of a server built from Lego and early reluctance to adopt advertising models, helped it become the leading search engine it is today, known for its mission to make the world's information universally accessible and useful."""
print(summarizer(text, max_length=50, min_length=10, do_sample=False))

# Translation with a specified model
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("My name is test program in python."))

# Zero-Shot Classification with a specified model
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zero_shot_classifier("gmply is an education company based in CA.", candidate_labels=["technology", "education", "politics"]))

#example-3
from transformers import pipeline

# Text Generation with a specified model and prompt for a story
text_generator = pipeline("text-generation", model="gpt2")

# Prompt for generating a story
prompt = "Write a short story about rain and snow."

# Generate the story
generated_texts = text_generator(prompt, max_length=200, num_return_sequences=1, truncation=True)

# Print the generated stories
for i, text in enumerate(generated_texts):
    print(f"Story {i + 1}:\n{text['generated_text']}\n")

#example-4 
from transformers import pipeline

# Named Entity Recognition with a specified model and additional parameters
ner_pipeline = pipeline(
    "ner", 
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",  # Combine adjacent entities of the same type
    grouped_entities=True  # Group entities by type
)

text = "Seattle, the largest city in the state of Washington, is renowned for its vibrant cultural scene, technological innovation, and stunning natural beauty. Located between Puget Sound and Lake Washington, Seattle is surrounded by water, mountains, and evergreen forests, giving it a unique blend of urban and natural environments. The city is a hub for major tech companies, including Amazon and Microsoft, which have significantly influenced its growth and development"

# Run NER pipeline on the input text
results = ner_pipeline(text)

# Print the results
for entity in results:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")

