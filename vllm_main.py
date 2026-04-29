import base64
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

app = FastAPI(title="Qwen3.5-9B reasoning-aware API")

# Initialize the client pointing to your local vLLM instance
# vLLM is running on 8000, this script runs on 8080
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

MODEL_NAME = "Qwen/Qwen3.5-9B"

@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail.")
):
    try:
        # 1. Read the image file and encode to base64
        image_bytes = await image.read()
        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        
        # 2. Automatically detect MIME type (handles webp, png, jpg)
        mime_type = image.content_type or "image/jpeg"
        data_url = f"data:{mime_type};base64,{base64_img}"

        # 3. Call the local vLLM engine
        # We use a higher max_tokens because reasoning models need space to "think"
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=2048,
            temperature=0.2
        )

        # 4. Extract the data
        # Qwen3.5/3.6 outputs reasoning_content if --reasoning-parser qwen3 is enabled
        message = response.choices[0].message
        message_dict = message.model_dump()
        
        reasoning = message_dict.get("reasoning_content")
        content = message_dict.get("content")

        return {
            "status": "success",
            "model": MODEL_NAME,
            "analysis": {
                "thoughts": reasoning if reasoning else "Direct output (no reasoning recorded)",
                "response": content if content else "No final response generated (check token limits)"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Host 0.0.0.0 makes the API accessible to your mobile app on the same network
    uvicorn.run(app, host="0.0.0.0", port=8080)