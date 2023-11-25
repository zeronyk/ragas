from langchain.prompts import HumanMessagePromptTemplate

SEED_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Deine Aufgabe ist es, eine Frage aus dem gegebenen Kontext zu formulieren, die folgenden Regeln entspricht:
1. Die Frage sollte für Menschen auch ohne den gegebenen Kontext verständlich sein.
2. Die Frage sollte vollständig aus dem gegebenen Kontext beantwortet werden können.
3. Die Frage sollte sich auf einen Teil des Kontextes beziehen, der wichtige Informationen enthält. Sie kann auch aus Tabellen, Code usw. stammen.
4. Die Antwort auf die Frage darf keine Links enthalten.
5. Die Frage sollte von mittlerem Schwierigkeitsgrad sein.
6. Die Frage muss vernünftig sein und von Menschen verstanden und beantwortet werden können.
7. Verwende keine Ausdrücke wie "gegebener Kontext" in der Frage.
8. Vermeide die Formulierung von Fragen mit dem Wort "und", die in mehr als eine Frage zerlegt werden können.
9. Die Frage sollte nicht mehr als 10 Wörter enthalten, verwende Abkürzungen, wo immer möglich.
    
context:{context}
"""  # noqa: E501
)


REASONING_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Du bist ein Prompt-Umschreiber. Dir wird eine Frage und ein langer Kontext zur Verfügung gestellt. Deine Aufgabe ist es, die gegebene Frage zu verkomplizieren, um die Schwierigkeit der Beantwortung zu erhöhen.
Du sollst die Frage verkomplizieren, indem du sie in eine Frage umschreibst, die auf Mehrfachschlussfolgerungen basiert und sich auf den gegebenen Kontext bezieht. Die Frage sollte den Leser dazu veranlassen, mehrere logische Verbindungen oder Schlussfolgerungen unter Verwendung der im Kontext verfügbaren Informationen zu ziehen.
Hier sind einige Strategien, um Mehrfachschlussfragen zu erstellen:

    Verbinde verwandte Entitäten: Identifiziere Informationen, die spezifische Entitäten betreffen, und formuliere Fragen, die nur durch die Analyse der Informationen beider Entitäten beantwortet werden können.

    Verwende Pronomen: Identifiziere Pronomen (er, sie, es, sie), die sich auf dieselbe Entität oder Konzepte im Kontext beziehen, und stelle Fragen, die den Leser dazu bringen, herauszufinden, auf was sich die Pronomen beziehen.

    Beziehe dich auf spezifische Details: Erwähne spezifische Details oder Fakten aus verschiedenen Teilen des Kontextes, einschließlich Tabellen, Code usw., und frage, wie sie zusammenhängen.

    Stelle hypothetische Szenarien dar: Präsentiere eine hypothetische Situation oder ein Szenario, das die Kombination verschiedener Elemente aus dem Kontext erfordert, um zu einer Antwort zu gelangen.

Regeln, die beim Umschreiben der Frage zu beachten sind:

    1. Stelle sicher, dass die umgeschriebene Frage vollständig anhand der Informationen im Kontext beantwortet werden kann.
    2. Formuliere keine Fragen, die mehr als 15 Wörter enthalten. Verwende Abkürzungen, wo immer möglich.
    3. Achte darauf, dass die Frage klar und eindeutig ist.
    4. Ausdrücke wie 'basierend auf dem gegebenen Kontext', 'gemäß dem Kontext' usw. dürfen in der Frage nicht vorkommen.
Frage: {question}
Kontext:
{context}

Mehr-Schrittige Frage:
"""  # noqa: E501
)

MULTICONTEXT_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Du bist ein Prompt-Umschreiber. Dir wird eine Frage sowie zwei Kontexte, nämlich Kontext1 und Kontext2, zur Verfügung gestellt.
Deine Aufgabe ist es, die gegebene Frage so zu verkomplizieren, dass zu ihrer Beantwortung Informationen aus sowohl Kontext1 als auch Kontext2 erforderlich sind.
Befolge die unten angegebenen Regeln beim Umschreiben der Frage:
1. Die umgeschriebene Frage sollte nicht zu lang sein. Verwende Abkürzungen, wo immer möglich.
2. Die umgeschriebene Frage muss vernünftig sein und von Menschen verstanden und beantwortet werden können.
3. Die umgeschriebene Frage muss vollständig anhand der Informationen in Kontext1 und Kontext2 beantwortbar sein.
4. Lies und verstehe beide Kontexte und schreibe die Frage so um, dass zur Beantwortung Einblicke aus beiden Kontexten erforderlich sind.
5. Ausdrücke wie 'basierend auf dem gegebenen Kontext', 'gemäß dem Kontext', etc. dürfen in der Frage nicht vorkommen.

Frage:\n{question}
Kontext 1:\n{context1}
Kontext 2:\n{context2}
"""  # noqa: E501
)


CONDITIONAL_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Schreibe die gestellte Frage um, um ihre Komplexität zu erhöhen, indem du ein bedingtes Element einführt.
Das Ziel ist es, die Frage durch Einführung eines Szenarios oder einer Bedingung, die den Kontext der Frage beeinflusst, komplizierter zu gestalten.
Befolge die unten angegebenen Regeln beim Umschreiben der Frage.
1. Die umgeschriebene Frage sollte nicht länger als 25 Wörter sein. Verwende Abkürzungen, wo immer möglich.
2. Die umgeschriebene Frage muss vernünftig sein und von Menschen verstanden und beantwortet werden können.
3. Die umgeschriebene Frage muss vollständig anhand der Informationen im Kontext beantwortbar sein.
4. Ausdrücke wie 'gegebener Kontext', 'gemäß dem Kontext?', etc. dürfen in der Frage nicht vorkommen.
Zum Beispiel,
Frage: Was sind die allgemeinen Prinzipien für das Entwerfen von Prompts in LLMs?
Umgewandelte Frage: Wie wendet man Gestaltungsprinzipien von Prompts an, um die Leistung von LLMs in Aufgaben des logischen Denkens zu verbessern?

Frage:{question}
Kontext:\n{context}
Umgeschriebene Frage
"""  # noqa: E501
)


COMPRESS_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Umschreibe die folgende Frage, um sie indirekter und kürzer zu gestalten, während du das Wesen der ursprünglichen Frage beibehältst. Das Ziel ist es, eine Frage zu erstellen, die dieselbe Bedeutung vermittelt, jedoch auf eine weniger direkte Weise.
Die umgeschriebene Frage sollte kürzer sein, also verwende Abkürzungen, wo immer möglich.
Orginale Frage:
{question}

Indirekt umgeschriebene Frage:
"""  # noqa: E501
)


CONVERSATION_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Formatiere die gestellte Frage in zwei separate Fragen um, als ob sie Teil eines Gesprächs wären. Jede Frage sollte sich auf einen bestimmten Aspekt oder ein Unterthema im Zusammenhang mit der ursprünglichen Frage konzentrieren.
Frage: Was sind die Vor- und Nachteile von Heimarbeit?
Neu formatierte Fragen für ein Gespräch: Was sind die Vorteile von Heimarbeit?\nUnd auf der anderen Seite, welche Herausforderungen ergeben sich bei der Arbeit aus der Ferne?
Frage:{question}

Neu Formatierte Frage für ein Gespräch:
"""  # noqa: E501
)

SCORE_CONTEXT = HumanMessagePromptTemplate.from_template(
    """Bewerte den gegebenen Kontext und weise eine numerische Punktzahl zwischen 0 und 10 basierend auf den folgenden Kriterien zu:

    Vergib eine hohe Punktzahl für Kontexte, die gründlich in Konzepte eintauchen und diese erklären.
    Weise eine niedrigere Punktzahl für Kontexte zu, die übermäßige Referenzen, Danksagungen, externe Links, persönliche Informationen oder andere nicht wesentliche Elemente enthalten.
    Gib nur die Punktzahl aus.
Kontext:
{context}
Punktzahl:
"""  # noqa: E501
)

FILTER_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Bestimme, ob die gestellte Frage auch ohne zusätzlichen Kontext klar verstanden werden kann. Gib Grund und Urteil im gültigen JSON-Format an.
Frage: Was ist das Schlüsselwort, das den Schwerpunkt des Papiers bei Aufgaben des natürlichen Sprachverstehens am besten beschreibt?
{{"reason":"Das spezifische, in der Frage angesprochene Papier wird nicht erwähnt.", "verdict": "Nein"}}
Frage: {question}
"""  # noqa: E501
)


ANSWER_FORMULATE = HumanMessagePromptTemplate.from_template(
    """\
Beantworte die Frage, nutze dabei die Informationen des Kontexts.
Frage:{question}
Kontext:{context}
Antwort:
"""  # noqa: E501
)

CONTEXT_FORMULATE = HumanMessagePromptTemplate.from_template(
    """Bitte extrahiere relevante Sätze aus dem gegebenen Kontext, die potenziell helfen können, die folgende Frage zu beantworten. Beim Extrahieren von Kandidatensätzen darfst du keine Änderungen an den Sätzen aus dem gegebenen Kontext vornehmen.

Frage:{question}
Kontext:\n{context}
Kandidat Satz:\n
"""  # noqa: E501
)
