from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from analyse import generate_dashboard

app = FastAPI()


@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return generate_dashboard()

from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")