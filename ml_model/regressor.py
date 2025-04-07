from sklearn.linear_model import LinearRegression

def get_regressor():
    #Gibt ein Klassifikation modell zurück
    #lineare Regression ist sinnvoll, wenn man z. B. wahrscheinlichkeiten oder Trends vorhersagen möchte
    #Wird später z. B. für vorhersage genutzt
    return LinearRegression()
