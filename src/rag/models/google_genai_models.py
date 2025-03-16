import os
import ast

from google import genai

from rag.models.prompts import (
    QUERY_AUGMENTATION_PROMPT_TEMPLATE,
    ENHANCE_SEARCH_PROMPT_TEMPLATE,
    ANSWER_GENERATION_PROMPT_TEMPLATE
)


class BaseGoogleModel:
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
    def __init__(self):
        super().__init__(prompt_template=QUERY_AUGMENTATION_PROMPT_TEMPLATE)


class EnhanceSearchModel(BaseGoogleModel):
    def __init__(self):
        super().__init__(prompt_template=ENHANCE_SEARCH_PROMPT_TEMPLATE)

    def generate(self, query: str, **kwargs):
        response = super().generate(query, **kwargs)
        response = ast.literal_eval(response[response.find("{"):response.find("}") + 1])

        summary = response.get("summary")
        questions = response.get("questions")

        return summary, questions


class AnswerGenerationModel(BaseGoogleModel):
    def __init__(self):
        super().__init__(prompt_template=ANSWER_GENERATION_PROMPT_TEMPLATE)
