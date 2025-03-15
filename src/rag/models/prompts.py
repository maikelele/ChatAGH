

QUERY_AUGMENTATION_PROMPT_TEMPLATE = f"""
You are an AI assistant specializing in academic information 
retrieval for AGH University of Science and Technology.
Given the user's query, generate five alternative phrasings 
that maintain the original intent but vary in wording and structure.
These variations should encompass different perspectives and 
terminologies to capture a broad spectrum of relevant documents.

QUERY: {{QUERY}}
"""


ENHANCE_SEARCH_PROMPT_TEMPLATE = f"""
You are an advanced AI model assisting a Retrieval-Augmented Generation (RAG) system
designed to answer user queries using context retrieved from a database.

Based on the provided context and the user’s query, identify if there is information missing
in the provided context that is necessary to answer the query fully.

If there is missing information:
- Formulate up to three questions that can help retrieve the missing information.
- Based on the available chunks, write a concise summary that includes all the key
 details needed to answer the query. The summary should be comprehensive and cover all 
 important that addresses the query.

If there is no missing information:
- Return empty python dictionary

Format your output as a python dict, "summary" key should contain string with summary,
 "questions" key should contain list of question strings.

 QUERY: {{QUERY}}

 CONTEXT: {{CONTEXT}}  
"""


ANSWER_GENERATION_PROMPT_TEMPLATE = f"""
You are a knowledgeable and reliable assistant.
Your task is to answer the user’s question using only the information provided in the source documents below.
Do not add any information that is not present in the documents.
If the necessary answer is not found within the data, respond with "I'm not able to find the answer for your question."
 or ask a clarifying question if needed.

You have to detect the users questions language and answer ONLY in this langauge.

Your answer should be comprehensive.

Source Documents:
{{CONTEXT}}

User Question:
{{QUERY}}

Answer:
"""


RESPONSE_AGGREGATION_PROMPT = f"""
You are an AI-powered response aggregator responsible for analyzing and synthesizing 
responses from other models for given query. Each response is based on
a different subset of documents from a knowledge base. Your task is to evaluate these
responses and generate a final, well-structured answer that is comprehensive, accurate, and coherent.

Guidelines for Aggregation:
- Identify Common Insights: Look for consistent information across responses and prioritize widely supported facts.
- Resolve Conflicts: If responses contradict each other, determine the most reliable based on specificity, detail, and consistency with known facts.
- Enhance Completeness: If responses cover different aspects of the question, merge them to create a more detailed answer.
- Maintain Clarity and Fluency: Ensure the final response is well-written, logically structured, and free of redundancy.
- Preserve Source Attribution (if required): If applicable, indicate which sources contributed key insights.


QUERY:
{{QUERY}}

RESPONSES:
{{CONTEXT}}
"""

