from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# Create the FastAPI application
app = FastAPI(
    title="NeurOps API",
    description="Intelligent business analytics powered by AI",
    version="1.0.0"
)

# CORS middleware — allows the frontend to talk to the backend
# Without this, browsers block requests between different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register our routes
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Welcome to NeurOps API"}