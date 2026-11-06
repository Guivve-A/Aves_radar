"""Pipeline de fine-tuning (aprendizaje por transferencia) de EcoAcoustic Sentinel.

Entrena una cabeza de clasificación local sobre los embeddings de BirdNET V2.4
usando las detecciones validadas por expertos del proyecto AudioMoth - CONDESAN.

Módulos principales:
    config          Rutas y constantes (sobrescribibles con variables de entorno).
    labels          Lectura de TablaFinal.xlsx y construcción de ventanas multietiqueta.
    select_subset   Selección estratificada de grabaciones bajo un presupuesto de descarga.
    drive_mirror    Construcción de la copia local (espejo) de los datos de Drive.
    embeddings      Extracción de embeddings y scores originales de BirdNET.
    dataset         Ensamblado de matrices X / Y / grupos para entrenamiento.
    heads           Cabezas de clasificación (prototipos coseno, lineal, MLP).
    train           Validación cruzada agrupada, entrenamiento final y exportación.
    report          Métricas, figuras y tablas LaTeX para el informe.
"""
