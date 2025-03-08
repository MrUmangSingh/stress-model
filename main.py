from fastapi import FastAPI
import uvicorn
from fastapi.responses import JSONResponse
from stress_predictor import stress_predictor
import numpy as np

app = FastAPI()


@app.post("/predict")
async def predict_stress(image: list):
    try:
        # Convert input list to NumPy array
        img = np.array(image, dtype=np.uint8)

        # Predict stress level
        predicted_class, output_data = stress_predictor(img)

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "output_data": output_data
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
