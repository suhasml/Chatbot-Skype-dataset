import warnings
warnings.filterwarnings("ignore")
import string
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json

import numpy as np
import random
from keras.models import load_model

words = []
classes = []
docs = []
ignore_words = string.punctuation
data_file = open("Train_Bot1.json").read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        docs.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for doc in docs:
    bag = []
    pat_words = doc[0]

    pat_words = [lemmatizer.lemmatize(word.lower()) for word in pat_words]

    for w in words:
        bag.append(1) if w in pat_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

model = load_model('chatbot.h5')



def clean_up_sequence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details = True):
    sentence_words = clean_up_sequence(sentence)

    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details = False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]

    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability":str(r[1])})
    
    return return_list

def getResponse(ints, intents_json):
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

