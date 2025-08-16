"""
Setup script for RestorAI MVP.
Makes the package installable with pip for easy distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "gradio>=4.7.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.25.0",
        "pydantic>=2.5.0",
        "aiofiles>=23.2.0",
        "python-multipart>=0.0.6",
        "click>=8.1.0",
        "tqdm>=4.66.0",
        "requests>=2.31.0",
    ]

setup(
    name="restorai-mvp",
    version="1.0.0",
    description="""
    A small, production-ready MVP for image & video restoration/enhancement. Web (React) + Gradio + PyQt on a FastAPI backend with modular pipelines (Real-ESRGAN, GFPGAN/CodeFormer, RIFE/EDVR), batch jobs, safety, A/B compare, and export.
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Justin21523",
    author_email="b0979865617@gmail.com",
    url="https://github.com/Justin21523/restoreAI-studio",
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # Dependencies
    install_requires=requirements,
    python_requires=">=3.8",
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "torchvision>=0.16.0+cu118",
        ],
        "models": [
            "realesrgan>=0.3.0",
            "gfpgan>=1.3.8",
            "basicsr>=1.4.2",
        ],
    },
    # Entry points for command line usage
    entry_points={
        "console_scripts": [
            "restorai=run:main",
            "restorai-ui=run:launch_gradio_ui",
            "restorai-api=run:launch_api_server",
            "restorai-cli=run:launch_cli",
            "restorai-download=scripts.download_models:main",
        ],
    },
    # Package data
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
        "ui": ["gradio/*.html", "gradio/*.css", "gradio/*.js"],
        "scripts": ["*.py"],
    },
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Keywords
    keywords="ai, image-restoration, super-resolution, face-enhancement, computer-vision, deep-learning",
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/restorai-mvp/issues",
        "Source": "https://github.com/your-username/restorai-mvp",
        "Documentation": "https://github.com/your-username/restorai-mvp/wiki",
    },
)
