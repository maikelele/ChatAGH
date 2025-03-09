import os

from ibm_watsonx_ai import APIClient, Credentials

ENV_PATH = "/Users/wnowogorski/PycharmProjects/ChatAGH/POC/config/.env"

def watsonx_client():
    credentials = Credentials(
        url=os.environ.get("WATSONX_URL"),
        api_key=os.environ.get("WATSONX_API_KEY"),
    )
    client = APIClient(credentials=credentials, space_id=os.environ.get("WATSONX_SPACE_ID"))

    return client

