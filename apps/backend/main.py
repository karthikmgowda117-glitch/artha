from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- THIS WAS MISSING
import uvicorn
from dotenv import load_dotenv
import os

# 1. Load environment variables
load_dotenv() 

# 2. Import your agent logic
from agents import ArthaDecisionEngine, IntelSignal, TradeAction

# 3. Initialize FastAPI
app = FastAPI(title="Artha v2.0: The Autonomous Indian Wealth Command Center")

# 4. Enable CORS so Karthik's JavaScript can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Initialize the Engine
engine = ArthaDecisionEngine()

@app.post("/analyze", response_model=TradeAction)
async def analyze_signal(signal: IntelSignal):
    try:
        # Triggers the adversarial debate (Bull vs Bear)
        decision = await engine.generate_action(signal)
        return decision
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)