from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass

import numpy as np
from datasets import Dataset
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM

CONTEXT_RECALL_RA = HumanMessagePromptTemplate.from_template(
    """
Gegeben einen Kontext und eine Antwort, analysiere jeden Satz in der Antwort und klassifiziere, ob der Satz dem gegebenen Kontext zugeordnet werden kann oder nicht. Gib das Ergebnis im JSON-Format mit Begründung aus.

Frage: Was können Sie mir über Albert Einstein erzählen?
Kontext: Albert Einstein (14. März 1879 – 18. April 1955) war ein in Deutschland geborener theoretischer Physiker, der weithin als einer der größten und einflussreichsten Wissenschaftler aller Zeiten angesehen wird. Bekannt wurde er vor allem durch die Entwicklung der Relativitätstheorie, er leistete jedoch auch wichtige Beiträge zur Quantenmechanik und war somit eine zentrale Figur bei der revolutionären Neugestaltung des wissenschaftlichen Verständnisses der Natur, die die moderne Physik in den ersten Jahrzehnten des zwanzigsten Jahrhunderts vollbrachte. Seine Massen-Energie-Äquivalenzformel E = mc2, die sich aus der Relativitätstheorie ergibt, wurde als "die berühmteste Gleichung der Welt" bezeichnet. Er erhielt 1921 den Nobelpreis für Physik "für seine Verdienste um die theoretische Physik und insbesondere für seine Entdeckung des photoelektrischen Effekts", ein entscheidender Schritt in der Entwicklung der Quantentheorie. Seine Arbeit ist auch bekannt für ihren Einfluss auf die Wissenschaftsphilosophie. In einer Umfrage aus dem Jahr 1999 unter 130 führenden Physikern weltweit von der britischen Zeitschrift Physics World wurde Einstein zum größten Physiker aller Zeiten gewählt. Seine intellektuellen Leistungen und seine Originalität haben Einstein zum Synonym für Genie gemacht.
Antwort: Albert Einstein, geboren am 14. März 1879, war ein in Deutschland geborener theoretischer Physiker, der weithin als einer der größten und einflussreichsten Wissenschaftler aller Zeiten angesehen wird. Er erhielt den Nobelpreis für Physik 1921 "für seine Verdienste um die theoretische Physik. Er veröffentlichte 1905 4 Arbeiten. Einstein zog 1895 in die Schweiz.
Klassifikation:
[
{{
"statement_1":"Albert Einstein, geboren am 14. März 1879, war ein in Deutschland geborener theoretischer Physiker, weithin als einer der größten und einflussreichsten Wissenschaftler aller Zeiten angesehen.",
"reason": "Das Geburtsdatum von Einstein wird im Kontext deutlich erwähnt.",
"Attributed": "Ja"
}},
{{
"statement_2":"Er erhielt 1921 den Nobelpreis für Physik 'für seine Verdienste um die theoretische Physik.",
"reason": "Der exakte Satz ist im gegebenen Kontext vorhanden.",
"Attributed": "Ja"
}},
{{
"statement_3": "Er veröffentlichte 1905 4 Arbeiten.",
"reason": "Es gibt keine Erwähnung der von ihm geschriebenen Arbeiten im gegebenen Kontext.",
"Attributed": "Nein"
}},
{{
"statement_4":"Einstein zog 1895 in die Schweiz.",
"reason": "Es gibt keine stützenden Beweise dafür im gegebenen Kontext.",
"Attributed": "Nein"
}}
]

Frage: Wer hat den ICC Weltcup 2020 gewonnen?
Kontext: Wer gewann den ICC Men's T20 World Cup 2022?
Der ICC Men's T20 World Cup 2022, der vom 16. Oktober bis zum 13. November 2022 in Australien stattfand, war die achte Ausgabe des Turniers. Ursprünglich für 2020 geplant, wurde es aufgrund der COVID-19-Pandemie verschoben. England ging als Sieger hervor und besiegte Pakistan im Finale mit fünf Wickets, um ihren zweiten ICC Men's T20 World Cup-Titel zu gewinnen.
Antwort: England
Klassifikation:
[
{{
"statement_1":"England gewann den ICC Men's T20 World Cup 2022.",
"reason": "Aus dem Kontext geht klar hervor, dass England Pakistan besiegte, um den Weltcup zu gewinnen.",
"Attributed": "Ja"
}}
]

Frage: {question}
Kontext:{context}
Antwort:{answer}
Klassifikation:
"""  # noqa: E501
)


@dataclass
class ContextRecall(MetricWithLLM):

    """
    Estimates context recall by estimating TP and FN using annotated answer and
    retrieved context.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_recall"
    evaluation_mode: EvaluationMode = EvaluationMode.qcg
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        dataset: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list:
        prompts = []
        question, ground_truths, contexts = (
            dataset["question"],
            dataset["ground_truths"],
            dataset["contexts"],
        )

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for qstn, gt, ctx in zip(question, ground_truths, contexts):
                gt = "\n".join(gt) if isinstance(gt, list) else gt
                ctx = "\n".join(ctx) if isinstance(ctx, list) else ctx
                human_prompt = CONTEXT_RECALL_RA.format(
                    question=qstn, context=ctx, answer=gt
                )
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            responses: list[list[str]] = []
            results = self.llm.generate(
                prompts,
                n=1,
                callbacks=batch_group,
            )
            responses = [[i.text for i in r] for r in results.generations]
            scores = []
            for response in responses:
                pattern = "\[\s*\{.*?\}(\s*,\s*\{.*?\})*\s*\]"
                match = re.search(pattern, response[0].replace("\n", ""))
                if match:
                    response = eval(response[0])
                    denom = len(response)
                    numerator = sum(
                        item.get("Attributed").lower() == "yes" for item in response
                    )
                    scores.append(numerator / denom)
                else:
                    scores.append(np.nan)

        return scores


context_recall = ContextRecall()
