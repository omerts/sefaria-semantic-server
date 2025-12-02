"""
FastAPI Application - Torah Source Finder API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .search import router as search_router
from .pipeline import router as pipeline_router
from .books import router as books_router


app = FastAPI(
    title="Torah Source Finder",
    description="Torah source search system based on Sefaria",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router, prefix="/api", tags=["search"])
app.include_router(pipeline_router, prefix="/api", tags=["pipeline"])
app.include_router(books_router, prefix="/api", tags=["books"])


@app.get("/")
async def root():
    """Home page"""
    return {
        "message": "Torah Source Finder API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search",
            "health": "/api/health",
            "pipeline": "/api/pipeline",
            "delete": "/api/delete",
            "books": "/api/books",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
