from fastapi import FastAPI
import uvicorn
import pandas as pd
import json
import statistics
from datetime import datetime
import time
#------------------------------------------------
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

app = FastAPI()


# uvicorn api:app --reload


@app.get("/")
async def root():
    return {"message": "ONLINE"}


@app.get("/sentiments/{key_word}/{start_date}/{end_date}")
async def sentiments(key_word, start_date, end_date):
    df = pd.read_csv("tweet.csv", sep=",")
    df_target = df[df['body'].str.contains(f"{key_word}")]
    start_date = start_date.replace("-","/")
    end_date = end_date.replace("-","/")
    start_date = int(time.mktime(datetime.strptime(start_date, "%d/%m/%Y").timetuple()))
    end_date = int(time.mktime(datetime.strptime(end_date, "%d/%m/%Y").timetuple()))


    df_target = df_target[ (df_target["post_date"] >= start_date) & (df_target["post_date"] <= end_date)]


    result = df_target.to_json(orient="split")
    parsed = json.loads(result)


    #--------- SPACY ---------#
    list_polarity = []

    for element in df_target['body'].values:
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('spacytextblob')
        doc = nlp(element)
        list_polarity.append(doc._.polarity)
    try:
        round_polarity = round(statistics.mean(list_polarity),3)
    except:
        round_polarity = "None"


    return f"Nombre de tweets trouvées pour '{key_word}' = {len(df_target)}",f"Moyenne des sentiments = {round_polarity}",f"Total comments = A FAIRE",f"Total retweet = A FAIRE",f"Total like = A FAIRE", parsed




#1 : URL : mettre un key word (citroen) => la page devra afficher nb likes, commebt, et partage

@app.get("/like_comment_partage/{key_word}")
async def like_comment_partage(key_word):
    return f"pour le keyword {key_word}, on a XXX like, XXX comment ...."

























if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


