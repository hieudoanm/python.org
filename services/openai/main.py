from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use OpenRouter API key and endpoint
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1"

app = FastAPI(
    title="FastAPI + OpenRouter Chat API",
    description="""
A simple REST API for interacting with OpenRouter models via the OpenAI-compatible API.
Default model: **openai/gpt-oss-20b**
You can send prompts and get AI-generated responses.

**Base URL**: `/`
**Docs**: `/docs` (Swagger UI) or `/redoc` (ReDoc)
""",
    version="1.0.0",
    contact={
        "name": "Your Name",
        "url": "https://openrouter.ai",
        "email": "you@example.com",
    },
    tags_metadata=[
        {
            "name": "Chat",
            "description": "Endpoints for sending prompts and receiving AI responses.",
        },
        {
            "name": "System",
            "description": "Endpoints for API health checks and server information.",
        },
    ],
)


# Request body model
class PromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        example="Write a haiku about the sunrise.",
        description="The user prompt or question you want the AI to respond to.",
    )
    model: str = Field(
        default="openai/gpt-oss-20b",
        example="openai/gpt-oss-20b",
        description="The model to use. Must be a valid OpenRouter model ID.",
    )


@app.post(
    "/chat",
    summary="Send a prompt to the AI model",
    description="Send a user prompt to the AI model and get the generated response.",
    tags=["Chat"],
)
async def chat(req: PromptRequest):
    """
    Send a prompt to the specified AI model and get the generated response.
    - **prompt**: The text you want to send to the model.
    - **model**: The model ID to use (e.g., `openai/gpt-oss-20b`).
    """
    try:
        response = openai.ChatCompletion.create(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
        )
        return {
            "prompt": req.prompt,
            "response": response.choices[0].message["content"],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get(
    "/",
    summary="Check API status",
    description="Returns a simple message confirming that the API is running.",
    tags=["System"],
)
async def root():
    return {"message": "FastAPI + OpenRouter (gpt-oss-20b default) server is running"}
