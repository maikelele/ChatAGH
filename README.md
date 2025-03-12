# Chat AGH

## Contents
- [Overview](#overview)
- [Developer guide](#developer-guide)
- [Data sources and preparation](#data-sources-and-preparation)
- [Documentation (Architecture and Technologies)](#documentation-architecture-and-technologies)
- [Metrics and evaluation](#metrics-and-evaluation)
- [Future Improvements](#future-improvements)

## Overview
Chat AGH is a Retrieval-Augmented Generation (RAG) system designed to deliver accurate and relevant information about academic matters at AGH University of Science and Technology. The system aggregates data sourced from university websites, enabling it to provide comprehensive answers to a wide range of user inquiries, which may include topics related to admissions, faculty-specific information, campus facilities, events, etc.

## Developer guide

### The same for all OS
Clone repository
```
git clone https://github.com/witoldnowogorski/ChatAGH
```
Create a new virtual environment
```
python3 -m venv chat_agh
```
### For linux/mac
Activate the environment
```
source chat_agh/bin/activate
```
### For windows
Move to next directory
```
cd chat_agh
```

Activate the enviroment
```
.\Scripts\activate
```
### The same for all OS
Install requirements
```
pip install -r requirements.txt
```
Run streamlit app
```
streamlit run src/streamlit_app.py
```

## Data sources and preparation

## Documentation (Architecture and Technologies)

## Metrics and evaluation

## Future Improvements
