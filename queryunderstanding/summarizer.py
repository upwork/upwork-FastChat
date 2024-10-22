from .retriever import Context
from .config.constants import SUMMARIZATION_LLM
from .utils import load_prompt, llm_client
from logging import getLogger

logger = getLogger(__name__)


class ResultsSummarizer:
    def summarize(self, context: Context) -> str:
        prompt_template = context.parameters[
            "results_summarizer_prompt"
        ] or load_prompt("results_summarization.txt")
        prompt = prompt_template.format(
            results="\n".join(context.objects["results"]),
            job=context.objects["job"],
            freelancer_names=", ".join(
                [freelancer["name"] for freelancer in context.objects["freelancers"]]
            ),
        )
        logger.info(f"Summarizing results:\n{prompt}")
        response = llm_client.chat.completions.create(
            model=SUMMARIZATION_LLM,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
