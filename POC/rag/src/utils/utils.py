import json
import time
import functools

from langchain_core.documents import Document


def load_from_json(path: str, max_docs: int = None):
    with open(path) as f:
        data = json.load(f)

    documents = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data][:max_docs]

    return documents


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
