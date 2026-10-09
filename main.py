from fastapi import FastAPI, HTTPException
from data_tools import get_clean_table, run_analysis, search_series

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "hello"}

@app.get("/double")
def double(number: int):
    return {"result": number * 2}

@app.get("/add")
def add(a: int, b: int):
    return {"result": a + b}

@app.get("/data")
def data(codes: str):
    code_list = codes.split(",")
    table, notes = get_clean_table(code_list)
    return {
        "months": len(table),
        "start": str(table.index[0].date()),
        "end": str(table.index[-1].date()),
        "notes": notes,
        "columns": list(table.columns),
    }

@app.get("/analyze")
def anayze(codes: str, target: str, start: str | None = None, end: str | None = None):
    try:
        return run_analysis(codes.split(","), target, start, end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/search")
def search(query: str):
    return search_series(query)

@app.post("/explain")
def explain(payload: dict):
    keep = ["target", "period", "selected_variables", "model",
            "coefficients", "vif", "diagnostics", "flags"]
    summary = {k: payload[k] for k in keep if k in payload}

    reply = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You explain regression results to a beginner in plain English. Be honest about weak results and warnings. Use short paragraphs, no jargon without a one-line explanation."},
            {"role": "user", "content": "Explain these results:\n" + json.dumps(summary)},
        ],
    )
    return {"explanation": reply.choices[0].message.content}