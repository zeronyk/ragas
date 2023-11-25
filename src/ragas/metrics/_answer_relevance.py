from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import trace_as_chain_group
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.embeddings.base import embedding_factory
from ragas.exceptions import OpenAIKeyNotFound
from ragas.metrics.base import EvaluationMode, MetricWithLLM

if t.TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManager

    from ragas.embeddings.base import RagasEmbeddings


QUESTION_GEN = HumanMessagePromptTemplate.from_template(
    """
Erstelle eine Frage für die gegebene Antwort.
Antwort:\nDie PSLV-C56-Mission ist geplant, am Sonntag, den 30. Juli 2023 um 06:30 IST / 01:00 UTC gestartet zu werden. Der Start erfolgt vom Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, Indien.
Frage: Wann ist der geplante Starttermin und die Uhrzeit für die PSLV-C56-Mission, und von wo aus wird sie gestartet?

Antwort:{answer}
Frage:
"""  # noqa: E501
)


@dataclass
class AnswerRelevancy(MetricWithLLM):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    batch_size: int
        batch size for evaluation
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    evaluation_mode: EvaluationMode = EvaluationMode.qa
    batch_size: int = 15
    strictness: int = 3
    embeddings: RagasEmbeddings = field(default_factory=embedding_factory)

    def init_model(self):
        super().init_model()

        if isinstance(self.embeddings, OpenAIEmbeddings):
            if self.embeddings.openai_api_key == "no-key":
                raise OpenAIKeyNotFound

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        questions, answers = dataset["question"], dataset["answer"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            prompts = []
            for ans in answers:
                human_prompt = QUESTION_GEN.format(answer=ans)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            results = self.llm.generate(
                prompts,
                n=self.strictness,
                callbacks=batch_group,
            )
            results = [[i.text for i in r] for r in results.generations]

            scores = []
            for question, gen_questions in zip(questions, results):
                cosine_sim = self.calculate_similarity(question, gen_questions)
                scores.append(cosine_sim.mean())

        return scores

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        )
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )


answer_relevancy = AnswerRelevancy()
