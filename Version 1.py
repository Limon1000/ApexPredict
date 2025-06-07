import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox

# Configurar FastF1 para usar cache y evitar descargas repetidas
fastf1.Cache.enable_cache('cache')

# Función para cargar datos y entrenar el modelo simple
def entrenar_modelo():
    # Cargar sesión 2025 
    session = fastf1.get_session(2025, 'Monaco', 'Q')
    session.load()
    
    # Extraer datos: piloto y tiempo de vuelta en segundos
    laps = session.laps
    laps = laps.pick_quicklaps()
    data = []
    for index, lap in laps.iterlaps():
        # Solo vueltas válidas
        if lap['IsPersonalBest']:
            data.append({
                'Driver': lap['Driver'],
                'LapTime': lap['LapTime'].total_seconds()
            })
    df = pd.DataFrame(data)
    
    # Convertir nombres de pilotos a números para este modelo
    df['DriverCode'] = pd.factorize(df['Driver'])[0]
    
    # Modelo simple que predice lap time solo con DriverCode
    X = df[['DriverCode']]
    y = df['LapTime']
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df

# Función para predecir con el modelo
def predecir_tiempo(driver_name, model, df):
    if driver_name not in df['Driver'].values:
        return None
    driver_code = df[df['Driver'] == driver_name]['DriverCode'].values[0]
    pred = model.predict([[driver_code]])
    return pred[0]

# Funcion al hacer click en botón predecir
def on_predecir():
    driver = entry_driver.get().strip()
    if not driver:
        messagebox.showwarning("Advertencia", "Introduce el código del piloto")
        return
    tiempo = predecir_tiempo(driver, modelo, df_datos)
    if tiempo:
        messagebox.showinfo("Predicción", f"Tiempo estimado para {driver}: {tiempo:.2f} segundos")
    else:
        messagebox.showerror("Error", f"Piloto {driver} no encontrado en los datos")

# Entrenar el modelo al inicio
modelo, df_datos = entrenar_modelo()

# Crear la ventana del tkinter
ventana = tk.Tk()
ventana.title("Predicción de tiempos F1 2025")

# Crear los widgets
label_driver = tk.Label(ventana, text="Código piloto (ej. VER):")
label_driver.pack(pady=5)

entry_driver = tk.Entry(ventana)
entry_driver.pack(pady=5)

btn_predecir = tk.Button(ventana, text="Predecir", command=on_predecir)
btn_predecir.pack(pady=10)

ventana.mainloop()
