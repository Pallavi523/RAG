import uvicorn
from config.settings import settings
from fastapi import FastAPI, Depends, HTTPException, Request, status
from api.hackrx import router as hackrx_router
from services.dependencies import initialize_services

app = FastAPI(title="RAG API", description="API for RAG project", version="1.0")

async def verify_token(request: Request):
    token = request.headers.get("Authorization")
    if not token or token != f"Bearer {settings.API_KEY}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return token

@app.on_event("startup")
async def startup_event():
    """Initialize services when the application starts."""
    print("Initializing services...")
    initialize_services()
    print("Services initialized successfully!")

app.include_router(hackrx_router, prefix="/api/v1", dependencies=[Depends(verify_token)])

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG API. Use the /docs endpoint to explore the API."}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )