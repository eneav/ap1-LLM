
# AP-LLM 
Ein KI-gestütztes Projekt zur Analyse, Prognose und Generierung von AP1-Prüfungsaufgaben für Fachinformatiker.

## Funktionen
- Konvertierung echter AP1-Prüfungen in JSON
- ML-gestützte Analyse (z. B. Klassifikation/Regression nach Thema)
- LLM-Generierung neuer, realistischer Prüfungsaufgaben



## Start (Schritt für Schritt)
1. `.env` mit eigenem api key anlegen
```env
OPENAI_API_KEY=sk-32432324v32XX
```

2. .venv erstellen und Pakete installieren:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Projekt starten mit Batch File oder manuell:
```bash
start_project.bat
```
Dadurch werden CSV-Dateien erzeugt (aus den JSON Files)

Diese werden für ML-Modelle(Klassifikation & Regression) genutzt
(Optional: Hauptfunktion = generierung neuer aufgaben)

---

## Erläuterung der Machine Learning Modelle

### `train_model.py` 
- Lädt alle JSON-Dateien aus `converted_json`
- Wandelt sie in ein gemeinsames CSV-Format (`training_data.csv`) um
- Diese Datei kann/wird von ML-Modellen genutzt (z.B. evaluate_model.py)

![visualisierung der training daten](bilder/trainingdata.png)


### `classifier.py`
- Liefert RandomForest-Klassifikationsmodell
```python
return RandomForestClassifier(n_estimators=100, random_state=42)
```

### `regressor.py`
- Liefert lineares Regressionsmodell (für die wahrscheinlichkeit der Themen)
```python
return LinearRegression()
```

### `evaluate_model.py`
- Lädt `training_data.csv`
- Vektorisiert die Fragen mit TF-IDF
- Trainiert das Klassifikationsmodell (z. B. für das Thema) 
- Bewertet Genauigkeit

### `generate_exam.py`
-Arbeitet mit GPT-4 über openai API. GPT-3.5 ist technisch möglich, liefert für diesen Use Case aber zu ungenaue Ergebnisse(upscaling bei bedarf möglich)
- Fragt das modell, um realistische neue Prüfungsaufgaben zu generieren (ergebnis in converted_json file) 

## Beispiel: Generierte Aufgabe 

![visualisierung der training daten](bilder/generierteAufgaben.png) 


---


## Erläuterung der Machine Learning Modelle

### `train_model.py` 
- Lädt alle JSON-Dateien aus `converted_json`
- Wandelt sie in ein gemeinsames CSV-Format (`training_data.csv`) um
- Diese Datei kann/wird von ML-Modellen genutzt (z.B. evaluate_model.py)

![visualisierung der training daten](bilder/trainingdata.png)


### `classifier.py`
- Liefert RandomForest-Klassifikationsmodell
```python
return RandomForestClassifier(n_estimators=100, random_state=42)
```

![klassifizierungsmodell](bilder/klassBewertung.png)

### `regressor.py`
- Liefert lineares Regressionsmodell (für die wahrscheinlichkeit der Themen)
```python
return LinearRegression()
```

### `evaluate_model.py`
- Lädt `training_data.csv`
- Vektorisiert die Fragen mit TF-IDF
- Trainiert das Klassifikationsmodell (z. B. für das Thema) 
- Bewertet Genauigkeit

### `generate_exam.py`
-Arbeitet mit GPT-4 über openai API. GPT-3.5 ist technisch möglich, liefert für diesen Use Case aber zu ungenaue Ergebnisse(upscaling bei bedarf möglich)
- Fragt das modell, um realistische neue Prüfungsaufgaben zu generieren (ergebnis in converted_json file) 

## Beispiel: Generierte Aufgabe 

![visualisierung der training daten](bilder/generierteAufgaben.png) 


---




