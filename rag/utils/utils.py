import os
import json

import time
import functools

from dotenv import load_dotenv
from langchain_core.documents import Document


def load_json_data(path: str):
    documents = []
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as json_file:
            file_data = json.load(json_file)
        i = 0
        try:
            i += 1
            document = Document(
                page_content=file_data["content"],
                metadata=file_data["metadata"]
            )
            documents.append(document)
        except Exception as e:
            print(f"Unable to read file: {file}, error: {e}")

    return documents

def load_env():
    env_path = os.path.join(os.getcwd(), 'config', '.env')
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def retry_on_exception(attempts=3, delay=1, backoff=10, exception=Exception):
    """
    A decorator to retry a function call if it raises a specified exception.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    if attempt == attempts:
                        raise
                    else:
                        print(f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff

        return wrapper

    return decorator
