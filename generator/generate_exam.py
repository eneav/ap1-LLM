import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_exam():
    prompt = (
    "Erstelle fünf realistische AP1-Prüfungsaufgaben für Fachinformatiker (FISI), "
    "Frühjahr 2026. Verwende typische Themen (z.B. Netzwerke, Datenbanken, OOP, ...), "
    "und formuliere für jede Aufgabe den Typ (offene Frage, multiple choice, Berechnung), "
    "sowie eine passende Antwort im folgenden JSON-Format:\n\n"
    "[\n"
    "  {\n"
    '    "fach": "FISI",\n'
    '    "jahr": 2026,\n'
    '    "saison": "Frühjahr",\n'
    '    "klausur_nr": 1,\n'
    '    "thema": "...",\n'
    '    "aufgabentyp": "...",\n'
    '    "frage": "...",\n'
    '    "antwort": "..."\n'
    "  },\n"
    "  ... (insgesamt 5 Aufgaben)\n"
    "]"
)



    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )

    json_string = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(json_string)
    except json.JSONDecodeError:
        print("fehler beim Parsen. Ausgabe:")
        print(json_string)
        return

    # weg zur datei (data\converted_json\generated_ap1_2026_03.json)

    output_path = os.path.join("data", "converted_json", "generated_ap1_2026_03.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"Generierte Aufgabe gespeichert unter: {output_path}")

if __name__ == "__main__":
    generate_exam()
