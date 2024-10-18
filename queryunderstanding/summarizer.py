from .config.constants import SUMMARIZATION_LLM
from .utils import load_prompt, llm_client


class ResultsSummarizer:
    def summarize(self, results: str) -> str:
        prompt_template = load_prompt("results_summarization.txt")
        prompt = prompt_template.format(results=results)
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
