"""Aprendizaje profundo sobre BirdNET V2.4 para los datos AudioMoth de CONDESAN.

Modulos:
    tflite_tools  modelo TFLite con salidas intermedias y lectura de pesos
    extract       audio -> embeddings, logits y mapas de activacion intermedios
    data          conjunto de ventanas con etiquetas, pliegues y pesos de muestreo
    metrics       metricas ponderadas, bootstrap por sesion
    heads         cabezas profundas sobre embeddings (PyTorch)
    backbone      replica en PyTorch de los bloques superiores de BirdNET
    train_heads   validacion cruzada de las cabezas
    finetune      ajuste fino de capas convolucionales
"""
