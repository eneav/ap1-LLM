#Lädt alle JSON files (aus converted_json)
#verbindet sie zu einem einheitlichen Data frame
#dieser datensatz für Training, Klassifikation und regression
import os
import json
import pandas as pd

def load_data_from_json(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            #jede Datei enthält mehrere Aufgaben, dann alles hinzufügen
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                data.extend(json.load(f))
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_data_from_json("data/converted_json")
    print("!DATEN ERFOLGREICH GELADEN!")
    print(df.head(5))  # Vorschau für erste 5 Zeilen
    df.to_csv("training_data.csv", index=False)  # speichert datensatz für spätere modelle
