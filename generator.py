import openai
import os
from dotenv import load_dotenv

load_dotenv()

class Generator:
    def __init__(self, model_name="gpt-4.5-preview", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

    def generate_answer(self, conversation, context, sources, max_tokens=2048):
        """
        Generate a freeform answer based on the conversation history, context, and sources.
        The answer should be as detailed as necessary without an arbitrary length limit.
        """
        messages = []

        # Instruct the assistant to be freeform.
        system_prompt = (
            "You are a highly knowledgeable assistant. "
            "Answer the user's question based solely on the provided context and conversation history. "
            "Your response should be freeform and as detailed as needed—do not limit your answer to a specific length. "
            "Include citations where applicable using the format [Source: <source>]."
        )
        messages.append({"role": "system", "content": system_prompt})

        # Provide the retrieved context.
        if context.strip():
            messages.append({"role": "system", "content": "Context: " + context})

        # Append the conversation history.
        for msg in conversation:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Include the source information.
        if sources:
            sources_text = "Sources provided: " + ", ".join(sources)
            messages.append({"role": "system", "content": sources_text})

        # Call the ChatCompletion API with a high max_tokens value.
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
