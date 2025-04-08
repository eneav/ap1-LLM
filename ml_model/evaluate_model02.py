# Nutzt die erzeugte CSV-Datei für Klassifikation (aus ergebnis von ml_model\train_model01.py)
#evaluate für vorhersage, zu welchem thema die Frage gehört

#er lädt die training_data.csv, welche zuvor in train_model01 generiert wurde
#daten werden in training/test gesplittet 
#klass. und regr. werden getestet 
#randomForestClassifier und linearRegression zeigen ihre Leistung

#ref:
#class: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#lin. regr: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

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
# TF-IDF=  Term Frequency-Inverse Document Frequency

#wörter werden in Vektoren umgewandelt, die Häufigkeit der Wörter in den Fragen wird berücksichtigt

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Klas. modell trainieren
model = RandomForestClassifier(random_state=42)
model.fit(X_train_vec, y_train)

# Vorhersage (basierend auf den testdatten | first run train_model01.py !!)
y_pred = model.predict(X_test_vec)

# Die auswertung zeigt, wie gut das modell ist 
print("Modell-Auswertung:\n")
print(classification_report(y_test, y_pred))



#auswerttung in a nutshell 

#precision = Wieiviel der vom Modell als z.b. Netzwerke vorhergesagten aufgaben waren auch wirklich netzwerke? 

#recall = wie viele aller tatsächlichen netzwerk aufgaben hat das modell richtig erkannt ? 

#f1-score = der mittelwert aus precision und recall (das ist besonders gut bei sehr unbalancierten aufgaben)

#support = anzahl der echten aufgaben pro klasse im testdatensatz 




#die warnungen sin dfür klassen ( z.b datenschutz o. it sicherheit) gibt es KEINE BEISPIELE ODER KEINE VORHERSAGEN im testdatensatz, also training_data.csv
