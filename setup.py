from setuptools import setup, find_packages

setup(
    name="obsidian-rag-chatgpt-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.95.2",
        "uvicorn>=0.15.0",
        "pydantic>=2.0.0",
        "python-dotenv>=0.19.0",
        "openai>=1.0.0",
        "faiss-cpu>=1.7.4",
        "PyYAML>=6.0.1",
        "python-dateutil>=2.8.2",
        "tiktoken>=0.5.0",
        "markdown>=3.3.0",
        "python-frontmatter>=1.0.0",
        "networkx>=3.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.10",
    ],
    python_requires=">=3.8",
)
