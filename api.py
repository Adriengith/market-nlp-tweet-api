from fastapi import FastAPI
import uvicorn
import pandas as pd
import json
import statistics
from datetime import datetime
#------------------------------------------------
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

app = FastAPI()





@app.get("/")
async def root():
    return {"message": "ONLINE"}


@app.get("/sentiments/{key_word}")
async def sentiments(key_word):
    df = pd.read_csv("tweet.csv", sep=",")
    df_target = df[df['body'].str.contains(f"{key_word}")]
    result = df_target.to_json(orient="split")
    parsed = json.loads(result)


    #--------- SPACY ---------#
    list_polarity = []

    for element in df_target['body'].values:
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('spacytextblob')
        doc = nlp(element)
        list_polarity.append(doc._.polarity)
        
    round_polarity = round(statistics.mean(list_polarity),3)
    #round_polarity = "HEROKU"


    return f"Nombre de tweets trouvées pour '{key_word}' = {len(df_target)}",f"Moyenne des sentiments = {round_polarity}", parsed


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
