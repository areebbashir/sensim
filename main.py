from fastapi import FastAPI
from pydantic import BaseModel
from bert import *

metadata=[
    {
        'name':'similarity',
        'description':'Finds the similarity between 2 sentences using their word vectors.'
        
    }
]

class CosSimilarityIn(BaseModel):
    text_1 : str
    text_2 : str
class CosSimilarityOut(BaseModel):
    score : float

app = FastAPI(title='Sentence_similarity',
              description='Deploying a basic word similairty based nlp model using FASTAPi',
              openapi_tags=metadata)

@app.get('/')
def home():
    return 'Welcome to the basic BERT based sentence similarity model'

@app.post('/similarity' , response_model=CosSimilarityOut,tags=['similarity'])
def similarity(sentences : CosSimilarityIn):
    score = cos_similarity(sentences.text_1, sentences.text_2)
    return {'score':score}

