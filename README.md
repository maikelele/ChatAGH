# Chat AGH

## Contents
- [Overview](#overview)
- [Developer guide](#developer-guide)
- [Data sources and preparation](#data-sources-and-preparation)
- [RAG implementation](#rag-implementation)
- [Metrics and evaluation](#metrics-and-evaluation)
- [Future Improvements](#future-improvements)

## Overview
Chat AGH is a Retrieval-Augmented Generation (RAG) system designed to deliver accurate and relevant information about academic matters at AGH University of Science and Technology. The system aggregates data sourced from university websites, enabling it to provide comprehensive answers to a wide range of user inquiries, which may include topics related to admissions, faculty-specific information, campus facilities, events, etc.

## Developer guide

### Clone repository
```
git clone https://github.com/witoldnowogorski/ChatAGH
cd ChatAGH
```

### Create environment
```
python3 -m venv chat_agh
```
Unix / MacOS
```
source chat_agh/bin/activate
```
Windows
```
cd chat_agh
.\Scripts\activate
```

### Install requirements
```
pip install -r requirements.txt
```

### Credentials
Add `.env` file with your credentials in config directory, you can find required credentials in `.env.template`

### Run streamlit app
```
streamlit run src/streamlit_app.py
```

## Data sources and preparation

## RAG implementation

## Metrics and evaluation

## Future Improvements
