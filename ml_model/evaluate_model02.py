# Nutzt die erzeugte CSV-Datei für Klassifikation (aus ergebnis von ml_model\train_model01.py)
# evaluate für vorhersage, zu welchem thema die Frage gehört
# UND: evaluiert zusätzlich das Regressionsmodell (z. B. Punkte, Schwierigkeit etc.)

# er lädt die training_data.csv, welche zuvor in train_model01 generiert wurde
# daten werden in training/test gesplittet 
# klass. und regr. werden getestet 
# RandomForestClassifier und LinearRegression zeigen ihre Leistung

# ref:
# class: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# lin. regr: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("training_data.csv")

print("\n---- KLASSIFIKATION -------n")

#auswertung der klass. ab hier

if "thema" in df.columns:

    # eingabe der x = frage und y = thema

    X_class = df["frage"]
    y_class = df["thema"]

    # aufteilung der trainings und testdaten | referenz dazu im repo https://github.com/eneav/ap1-LLM.git



    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    # Vektorisieren | Wandelt Text (Fragen) in Zahlen um, damit ein ML-Modell damit rechnen kann
    # TF-IDF=  Term Frequency-Inverse Document Frequency
    # wörter werden in Vektoren umgewandelt, die Häufigkeit der Wörter in den Fragen wird berücksichtigt

    vectorizer_c = TfidfVectorizer()
    X_train_vec_c = vectorizer_c.fit_transform(X_train_c)
    X_test_vec_c = vectorizer_c.transform(X_test_c)

    # Klas. modell trainieren

    model_class = RandomForestClassifier(random_state=42)
    model_class.fit(X_train_vec_c, y_train_c)

    # Vorhersage (basierend auf den testdaten | first run train_model01.py !!)

    y_pred_class = model_class.predict(X_test_vec_c)

    # Die auswertung zeigt, wie gut das modell ist 
    print("Modell Auswertung:\n")
    print(classification_report(y_test_c, y_pred_class, zero_division=0))

    # auswertung in a nutshell 

    # precision = Wieiviel der vom (klassifikations)Modell als z.b. Netzwerke vorhergesagten aufgaben waren auch wirklich netzwerke? 
    # recall = wie viele aller tatsächlichen netzwerk aufgaben hat das modell richtig erkannt ? 
    # f1-score = der mittelwert aus precision und recall (das ist besonders gut bei sehr unbalancierten aufgaben)
    # support = anzahl der echten aufgaben pro klasse im testdatensatz 
    # die warnungen sind für klassen (z. B. Datenschutz o. IT-Sicherheit), für die es KEINE BEISPIELE oder KEINE VORHERSAGEN im testdatensatz gab

else:
    print(" Spalte 'thema' fehlt - klassifikation wird übersprungen")

print("\n---------- REGRESSION -------\n")

# === REGRESSION ===
if "punkte" in df.columns:
    # x = frage, y = punkte (z. B. Schwierigkeit oder Bewertung)
    X_reg = df["frage"]
    y_reg = df["punkte"]

    # daten splitten
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Vektorisierung auch für Regression
    vectorizer_r = TfidfVectorizer()
    X_train_vec_r = vectorizer_r.fit_transform(X_train_r)
    X_test_vec_r = vectorizer_r.transform(X_test_r)

    # regr. trainieren
    model_reg = LinearRegression()
    model_reg.fit(X_train_vec_r, y_train_r)

    # Vorhersage
    y_pred_reg = model_reg.predict(X_test_vec_r)

    print("Regressions-Auswertung:\n")
    print(f"MAE:  {mean_absolute_error(y_test_r, y_pred_reg):.2f}")     # mittlerer absoluter Fehler
    print(f"RMSE: {root_mean_squared_error(y_test_r, y_pred_reg):.2f}")
                                                                                     # Wurzel aus mittlerem quadr. Fehler
    print(f"R²:   {r2_score(y_test_r, y_pred_reg):.2f}")               # Erklärte Varianz

else:
    print(" Spalte 'punkte' fehlt - Regression wird übersprungen.")
