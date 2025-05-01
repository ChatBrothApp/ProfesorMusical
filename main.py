from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Servidor de Profesor de Afinación funcionando!"}

@app.post("/detectar_nota/")
async def detectar_nota(audio: UploadFile = File(...)):
    contents = await audio.read()
    
    # Guardar temporalmente
    with open("temp_audio.wav", "wb") as f:
        f.write(contents)
    
    # Cargar audio
    y, sr = librosa.load("temp_audio.wav", sr=None)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    # Tomar frecuencia dominante
    f0_clean = f0[~np.isnan(f0)]
    
    if len(f0_clean) == 0:
        return {"error": "No se detectó tono."}
    
    frecuencia = np.median(f0_clean)
    nota = librosa.hz_to_note(frecuencia)
    
    return {"nota_detectada": nota}
