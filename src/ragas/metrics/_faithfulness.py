from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
from langchain.callbacks.manager import CallbackManager, trace_as_chain_group
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from ragas.metrics.base import EvaluationMode, MetricWithLLM
from ragas.utils import load_as_json

if t.TYPE_CHECKING:
    from datasets import Dataset


LONG_FORM_ANSWER_PROMPT = HumanMessagePromptTemplate.from_template(
    """\
Erstelle eine oder mehrere Aussagen aus jedem Satz in der gegebenen Antwort.

Frage: Wer war Albert Einstein und wofür ist er am bekanntesten?
Antwort: Er war ein in Deutschland geborener theoretischer Physiker, der weithin als einer der größten und einflussreichsten Physiker aller Zeiten anerkannt wird. Er war am besten bekannt für die Entwicklung der Relativitätstheorie, er leistete auch wichtige Beiträge zur Entwicklung der Quantenmechanik.
statements in JSON:
{{
"statements": [
"Albert Einstein wurde in Deutschland geboren.",
"Albert Einstein war am besten bekannt für seine Relativitätstheorie."
]
}}

Frage: Cadmiumchlorid ist in dieser Chemikalie leicht löslich, es wird auch wie genannt?
Antwort: Alkohol
statements in JSON:
{{
"statements": [
"Cadmiumchlorid ist leicht löslich in Alkohol."
]
}}

Frage: Waren Shahul und Jithin derselben Nationalität?
Antwort: Sie stammten aus verschiedenen Ländern.
statements as JSON:
{{
"statements": [
"Shahul und Jithin stammten aus verschiedenen Ländern."
]
}}

Frage:{question}
Antwort: {answer}
statements in JSON:"""  # noqa: E501
)


NLI_STATEMENTS_MESSAGE = HumanMessagePromptTemplate.from_template(
    """
Natürliche Sprachinferenz, 
Betrachten Sie den gegebenen Kontext und die folgenden Aussagen, und bestimmen Sie dann, ob sie durch die Informationen im Kontext gestützt werden. Geben Sie eine kurze Erklärung für jede Aussage ab, bevor Sie zu einem Urteil (Ja/Nein) kommen. Geben Sie am Ende ein abschließendes Urteil für jede Aussage in der angegebenen Reihenfolge und im vorgegebenen Format an. Weichen Sie nicht von dem festgelegten Format ab.
Schreibe unbedingt korrektes JSON, achte auf Kommata.
Kontext:
John ist Student an der XYZ Universität. Er studiert Informatik. Dieses Semester hat er sich für mehrere Kurse eingeschrieben, darunter Datenstrukturen, Algorithmen und Datenbankmanagement. John ist ein fleißiger Student und verbringt viel Zeit mit Studieren und Erledigen von Aufgaben. Er bleibt oft spät in der Bibliothek, um an seinen Projekten zu arbeiten.
statement_1: John hat sich auf Biologie spezialisiert.
statement_2: John besucht einen Kurs über Künstliche Intelligenz.
statement_3: John ist ein engagierter Student.
statement_4: John hat einen Teilzeitjob.
Antwort:
[
{{
"statement_1": "John hat sich auf Biologie spezialisiert.",
"reason": "Johns Hauptfach wird explizit als Informatik genannt. Es gibt keine Informationen, die darauf hindeuten, dass er sich auf Biologie spezialisiert hat.",
"verdict": "Nein"
}},
{{
"statement_2": "John besucht einen Kurs über Künstliche Intelligenz.",
"reason": "Der Kontext nennt die Kurse, für die John derzeit eingeschrieben ist, und Künstliche Intelligenz wird nicht erwähnt. Daher kann nicht gefolgert werden, dass John einen Kurs über KI besucht.",
"verdict": "Nein"
}},
{{
"statement_3": "John ist ein engagierter Student.",
"reason": "Im Kontext wird erwähnt, dass er viel Zeit mit Studieren und Erledigen von Aufgaben verbringt. Außerdem wird erwähnt, dass er oft spät in der Bibliothek bleibt, um an seinen Projekten zu arbeiten, was auf Engagement hindeutet.",
"verdict": "Ja"
}},
{{
"statement_4": "John hat einen Teilzeitjob.",
"reason": "Im Kontext gibt es keine Informationen darüber, dass John einen Teilzeitjob hat.",
"verdict": "Nein"
}}
]

Kontext:
Fotosynthese ist ein Prozess, der von Pflanzen, Algen und bestimmten Bakterien verwendet wird, um Lichtenergie in chemische Energie umzuwandeln.
statement_1: Antwort nicht im gegebenen Kontext gefunden
Antwort:
[
{{
"statement_4": "Antwort nicht im gegebenen Kontext gefunden",
"reason": "Der Kontext liefert nicht genügend Informationen, um die Gültigkeit der Aussage zu bestimmen.",
"verdict": "NULL"
}}
]


Kontext:
{context}
Aussagen:
{statements}
Answer:
"""  # noqa: E501
)


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"
    evaluation_mode: EvaluationMode = EvaluationMode.qac
    batch_size: int = 15

    def _score_batch(
        self: t.Self,
        ds: Dataset,
        callbacks: t.Optional[CallbackManager] = None,
        callback_group_name: str = "batch",
    ) -> list[float]:
        """
        returns the NLI score for each (q, c, a) pair
        """

        question, answer, contexts = ds["question"], ds["answer"], ds["contexts"]
        prompts = []

        with trace_as_chain_group(
            callback_group_name, callback_manager=callbacks
        ) as batch_group:
            for q, a in zip(question, answer):
                human_prompt = LONG_FORM_ANSWER_PROMPT.format(question=q, answer=a)
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)
            prompts = []
            for context, output in zip(contexts, result.generations):
                statements = load_as_json(output[0].text).get("statements", [])
                statements_str: str = "\n".join(
                    [f"statement_{i+1}: {st}" for i, st in enumerate(statements)]
                )
                contexts_str: str = "\n".join(context)
                human_prompt = NLI_STATEMENTS_MESSAGE.format(
                    context=contexts_str, statements=statements_str
                )
                prompts.append(ChatPromptTemplate.from_messages([human_prompt]))

            result = self.llm.generate(prompts, callbacks=batch_group)
            # write this into a fileo
            outputs = result.generations
            output_text = "\n".join([str(o) for o in outputs])  # Convert outputs to string
            file_path = "/home/hermel/Documents/Metis-Demo/optimizing_tt/faithful_debug/debug.txt"
            with open(file_path, 'a') as f:  # Open file in append mode
                f.write(output_text)
                f.write("\n") 
            verdict_score_map = {"ja": 1, "nein": 0, "null": np.nan}
            scores = []
            for output in outputs:
                output = load_as_json(output[0].text)
                output = output if output else []
                faithful_statements = sum(
                    verdict_score_map.get(dict.get("verdict", "").lower(), np.nan)
                    for dict in output
                )
                num_statements = len(output)
                if num_statements:
                    score = faithful_statements / num_statements
                else:
                    score = np.nan
                scores.append(score)

        return scores


faithfulness = Faithfulness()
