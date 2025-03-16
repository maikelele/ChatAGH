import os
import ast

from google import genai

from rag.models.prompts import (
    QUERY_AUGMENTATION_PROMPT_TEMPLATE,
    ENHANCE_SEARCH_PROMPT_TEMPLATE,
    ANSWER_GENERATION_PROMPT_TEMPLATE
)


class BaseGoogleModel:
    """
    A base class for interacting with Google's GenAI models.

    This class provides a common interface for generating content using a specified model
    and prompt template. It serves as a foundation for specialized models for query augmentation,
    search enhancement, and answer generation.

    Attributes:
        client (genai.Client): The Google API client initialized using the API key from environment variables.
        model (str): The name of the model used for content generation.
        prompt_template (str): A template used to format prompts for the model.

    Methods:
        _inference(contents):
            Sends a list of prompt contents to the model and returns the generated text.
        generate(query: str, **kwargs):
            Formats the prompt using the provided query and context, and generates a response.
    """

    def __init__(self, prompt_template, model_name="gemini-2.0-flash-001"):
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model_name
        self.prompt_template = prompt_template

    def _inference(self, contents):
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
        ).text

    def generate(self, query: str, **kwargs):
        context = kwargs.get("context", [])
        prompt = self.prompt_template.format(
            CONTEXT=context,
            QUERY=query
        )
        contents = [prompt]
        response = self._inference(contents)
        return response


class QueryAugmentationModel(BaseGoogleModel):
    """
    A model for query augmentation using Google's GenAI.

    This class leverages a predefined query augmentation prompt template to generate
    alternative phrasings for a user's query. It creates diverse query variants while preserving
    the original intent.

    Inherits from:
        BaseGoogleModel
    """

    def __init__(self):
        super().__init__(prompt_template=QUERY_AUGMENTATION_PROMPT_TEMPLATE)


class EnhanceSearchModel(BaseGoogleModel):
    """
    A model for enhancing search queries using Google's GenAI.

    This class utilizes an enhanced search prompt template to analyze the provided context and query.
    It identifies any missing information necessary to fully answer the query and, if found,
    generates a comprehensive summary along with up to three clarifying questions.

    Inherits from:
        BaseGoogleModel

    Methods:
        generate(query: str, **kwargs):
            Generates a response and post-processes it to extract a summary and a list of questions.
    """

    def __init__(self):
        super().__init__(prompt_template=ENHANCE_SEARCH_PROMPT_TEMPLATE)

    def generate(self, query: str, **kwargs):
        response = super().generate(query, **kwargs)
        response = ast.literal_eval(response[response.find("{"):response.find("}") + 1])
        summary = response.get("summary")
        questions = response.get("questions")
        return summary, questions


class AnswerGenerationModel(BaseGoogleModel):
    """
    A model for answer generation using Google's GenAI.

    This class employs an answer generation prompt template to produce comprehensive answers
    based solely on the provided source documents. It ensures that answers are derived only
    from the available context, and responds in the same language as the user query.

    Inherits from:
        BaseGoogleModel
    """

    def __init__(self):
        super().__init__(prompt_template=ANSWER_GENERATION_PROMPT_TEMPLATE)
