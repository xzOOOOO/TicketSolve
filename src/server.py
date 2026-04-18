import uvicorn
import sys
import os
from config import settings

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
)