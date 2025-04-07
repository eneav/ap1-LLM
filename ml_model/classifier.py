#Gibt ein Klassifikation modell zurück
#In unserem Fall: Random Forest 
#Wird später z. B. für vorhersage genutzt 

from sklearn.ensemble import RandomForestClassifier


def get_classifier():
    return RandomForestClassifier(n_estimators=100, random_state=42)
