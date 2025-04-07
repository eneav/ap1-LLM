# Nutzt die erzeugte CSV-Datei für Klassifikation
#evaluate für vorhersage, zu welchem thema die Frage gehört



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("training_data.csv")

# eingabe der x = frage und y = thema
X = df["frage"]
y = df["thema"]

# aufteilung der trainings und testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vektorisieren | Wandelt Text (Fragen) in Zahlen um, damit ein ML-Modell damit rechnen kann
# TF-IDF: Term Frequency-Inverse Document Frequency
#wörter werden in Vektoren umgewandelt, die Häufigkeit der Wörter in den Fragen wird berücksichtigt
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Klas. modell trainieren
model = RandomForestClassifier(random_state=42)
model.fit(X_train_vec, y_train)

# Vorhersage (basierend auf den testdatten)
y_pred = model.predict(X_test_vec)

# Auswertung, zeigt wie gut das Modell ist
print("Modell-Auswertung:\n")
print(classification_report(y_test, y_pred))
