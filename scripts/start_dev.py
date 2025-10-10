#!/usr/bin/env python3
"""
Development server startup script
"""
import uvicorn
from api.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level,
    )
