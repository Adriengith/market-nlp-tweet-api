from fastapi import FastAPI
import uvicorn
import pandas as pd
import json
import statistics
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from datetime import datetime


app = FastAPI()





@app.get("/")
async def root():
    return {"message": "Hello"}


@app.get("/sentiments/{key_word}")
async def sentiments(key_word):
    df = pd.read_csv("tweet.csv", sep=",")
    df_target = df[df['body'].str.contains(f"{key_word}")]
    result = df_target.to_json(orient="split")
    parsed = json.loads(result)

    #classer les phrases par rapport à leurs positivités 
    list_polarity = []

    # for element in df_target['body'].values:
    #     nlp = spacy.load('en_core_web_sm')
    #     nlp.add_pipe('spacytextblob')
    #     doc = nlp(element)
    #     list_polarity.append(doc._.polarity)
        
    #round_polarity = round(statistics.mean(list_polarity),3)
    round_polarity = [0,100]


    return f"Nombre de tweets trouvées pour '{key_word}' = {len(df_target)}",f"Moyenne des sentiments = {round_polarity}", parsed

@app.get("/test2")
async def test2():
    key_word = "test"
    return {"id": f"{key_word}"}

# @app.get("/students/id/{student_id}")
# async def get_student_by_id(student_id):
#     return {"student_id": student_id}


# @app.get("/students/name/{lastname}")
# async def get_student_by_name(lastname):
#     return {"lastname": lastname}


# @app.post("/students")
# async def create_student(student: Student):
#     student_id = str(COLLECTION_STUDENTS.insert_one(student.dict()).inserted_id)
#     # TODO : Envoyer un mail de confirmation
#     return {"student_id": student_id}


# @app.put("/students")
# async def update_student():
#     return "Not yet implemented"


@app.delete("/students/id/{id}")
async def delete_student_by_id(id):
    return {"id": id}


# @app.delete("/students/name/{lastname}")
# async def delete_student_by_name(lastname):
#     return {"lastname": lastname}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
