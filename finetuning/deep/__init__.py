"""Aprendizaje profundo sobre BirdNET V2.4 para los datos AudioMoth de CONDESAN.

Modulos:
    tflite_tools  modelo TFLite con salidas intermedias y lectura de pesos
    extract       audio -> embeddings, logits y mapas de activacion intermedios
    data          conjunto de ventanas con etiquetas y pesos de muestreo
    mixtures      banco de cantos validados atenuados sobre fondos de su misma sesion
    backbone      replica en PyTorch de los bloques superiores de BirdNET
    protocol      categorias de cada par, pliegues, metricas M1-M3 y reglas de decision
    train_ra      entrenamiento consciente de la revision: cabezas y capas convolucionales
    probes        sondas desde cero y verificador de dos etapas
    evaluate_ra   evaluacion final con bootstrap
    report        tablas y figuras del informe
"""
