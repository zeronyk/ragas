from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import load_as_json

CONTEXT_PRECISION = HumanMessagePromptTemplate.from_template(
    """\
Überprüfe, ob die Informationen im gegebenen Kontext nützlich sind, um die Frage zu beantworten.

Frage: Welche gesundheitlichen Vorteile hat grüner Tee?
Kontext:
Dieser Artikel erkundet die reiche Geschichte des Teeanbaus in China und verfolgt seine Wurzeln zurück zu den alten Dynastien. Er diskutiert, wie verschiedene Regionen ihre einzigartigen Teesorten und Brautechniken entwickelt haben. Der Artikel geht auch auf die kulturelle Bedeutung des Tees in der chinesischen Gesellschaft ein und wie er zu einem Symbol für Gastfreundschaft und Entspannung geworden ist.
Überprüfung:
{{"reason":"Der Kontext, obwohl informativ über die Geschichte und kulturelle Bedeutung des Tees in China, liefert keine spezifischen Informationen über die gesundheitlichen Vorteile von grünem Tee. Daher ist er nicht nützlich, um die Frage nach den gesundheitlichen Vorteilen zu beantworten.", "verdict":"Nein"}}

Frage: Wie funktioniert Photosynthese bei Pflanzen?
Kontext:
Photosynthese in Pflanzen ist ein komplexer Prozess, der mehrere Schritte umfasst. Dieses Papier erläutert, wie Chlorophyll in den Chloroplasten Sonnenlicht absorbiert, welches dann die chemische Reaktion antreibt, die Kohlendioxid und Wasser in Glukose und Sauerstoff umwandelt. Es erklärt die Rolle von Licht- und Dunkelreaktionen und wie ATP und NADPH während dieser Prozesse produziert werden.
Überprüfung:
{{"reason":"Dieser Kontext ist äußerst relevant und nützlich, um die Frage zu beantworten. Er geht direkt auf die Mechanismen der Photosynthese ein und erklärt die Schlüsselkomponenten und Prozesse, die daran beteiligt sind.", "verdict":"Ja"}}
Frage:{question}
Kontext:
{context}
Überprüfung:"""  # noqa: E501
)


@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_precision"
    evaluation_mode: EvaluationMode = EvaluationMode.qc
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        questions, contexts = dataset["question"], dataset["contexts"]
        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for qstn, ctx in zip(questions, contexts):
                human_prompts = [
                    ChatPromptTemplate.from_messages(
                        [CONTEXT_PRECISION.format(question=qstn, context=c)]
                    )
                    for c in ctx
                ]

                prompts.extend(human_prompts)

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]
            context_lens = [len(ctx) for ctx in contexts]
            context_lens.insert(0, 0)
            context_lens = np.cumsum(context_lens)
            grouped_responses = [
                responses[start:end]
                for start, end in zip(context_lens[:-1], context_lens[1:])
            ]
            scores = []

            for response in grouped_responses:
                response = [load_as_json(item) for item in sum(response, [])]
                response = [
                    int("yes" in resp.get("verdict", " ").lower())
                    if resp.get("verdict")
                    else np.nan
                    for resp in response
                ]
                denominator = sum(response) + 1e-10
                numerator = sum(
                    [
                        (sum(response[: i + 1]) / (i + 1)) * response[i]
                        for i in range(len(response))
                    ]
                )
                scores.append(numerator / denominator)

        return scores


context_precision = ContextPrecision()
