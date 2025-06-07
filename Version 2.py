import fastf1
from fastf1 import plotting
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import random

# Simulamos entrenamiento (usa tus propios datos históricos para hacerlo real)
def entrenar_modelo():
    data = pd.DataFrame({
        'driver': ['VER', 'HAM', 'LEC', 'ALO', 'SAI', 'NOR'],
        'team': ['Red Bull', 'Mercedes', 'Ferrari', 'Aston Martin', 'Ferrari', 'McLaren'],
        'position': [1, 2, 3, 4, 5, 6],
        'accident': [0, 0, 0, 1, 0, 0]
    })
    
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    data['driver_encoded'] = le_driver.fit_transform(data['driver'])
    data['team_encoded'] = le_team.fit_transform(data['team'])

    features = data[['driver_encoded', 'team_encoded']]
    target_pos = data['position']
    target_acc = data['accident']

    modelo_pos = RandomForestClassifier().fit(features, target_pos)
    modelo_acc = RandomForestClassifier().fit(features, target_acc)

    return modelo_pos, modelo_acc, le_driver, le_team

modelo_pos, modelo_acc, le_driver, le_team = entrenar_modelo()

# Función para predecir posición y accidente
def predecir_resultado(driver, team):
    try:
        driver_encoded = le_driver.transform([driver])[0]
        team_encoded = le_team.transform([team])[0]
    except:
        messagebox.showerror("Error", "Piloto o equipo no conocidos en el modelo.")
        return

    X = [[driver_encoded, team_encoded]]

    pred_pos = modelo_pos.predict(X)[0]
    pred_acc = modelo_acc.predict_proba(X)[0][1]  # Probabilidad de accidente

    mensaje = f"🔢 Posición estimada: {pred_pos}\n🚧 Probabilidad de accidente: {pred_acc:.2%}"
    messagebox.showinfo("Predicción", mensaje)

# Interfaz gráfica
ventana = tk.Tk()
ventana.title("Predicción F1 - Accidentes y Posición")

tk.Label(ventana, text="Piloto (ej. VER):").pack()
entrada_piloto = tk.Entry(ventana)
entrada_piloto.pack()

tk.Label(ventana, text="Equipo (ej. Red Bull):").pack()
entrada_equipo = tk.Entry(ventana)
entrada_equipo.pack()

btn_predecir = tk.Button(ventana, text="Predecir", command=lambda: predecir_resultado(
    entrada_piloto.get().strip(), entrada_equipo.get().strip()))
btn_predecir.pack(pady=10)

ventana.mainloop()
