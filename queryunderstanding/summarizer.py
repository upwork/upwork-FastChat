from .retriever import Context
from .config.constants import SUMMARIZATION_LLM
from .utils import load_prompt, llm_client


class ResultsSummarizer:
    def summarize(self, context: Context) -> str:
        prompt_template = context.parameters[
            "results_summarizer_prompt"
        ] or load_prompt("results_summarization.txt")
        prompt = prompt_template.format(
            results=context.objects["results"],
            job=context.objects["job"],
        )
        response = llm_client.chat.completions.create(
            model=SUMMARIZATION_LLM,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes the results.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
