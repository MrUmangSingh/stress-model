from fastapi import FastAPI
import uvicorn
from fastapi.responses import JSONResponse
from stress_predictor import stress_predictor, read_image_from_url
from pydantic import BaseModel as PydanticBaseModel, Field

app = FastAPI()


class URLRequest(PydanticBaseModel):
    url: str = Field(description="The image to be predicted")


@app.post("/predict")
async def predict_stress(request: URLRequest):
    try:
        img = read_image_from_url(request.url)
        output_data = stress_predictor(img)
        output_data = output_data.astype(float).tolist()
        output_data = output_data*100
        output_data = round(output_data, 2)
        return {
            "stress_level": output_data
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
