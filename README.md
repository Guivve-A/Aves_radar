# EcoAcoustic Sentinel

Desktop application for bird bioacoustic monitoring with a hybrid detection pipeline.
Aplicacion de escritorio para monitoreo bioacustico aviar con un pipeline hibrido.

## Español

### Descripcion

EcoAcoustic Sentinel combina:
- Biblioteca local de especies (few-shot por embeddings)
- Descubrimiento global con BirdNET

### Caracteristicas principales

- Flujo guiado en 4 pasos:
  1. Audio
  2. Zona
  3. Parametros
  4. Exportacion
- Estrategia hibrida:
  - Si hay hits validos de perfiles custom, se priorizan esos resultados.
  - Si no hay hits custom, se puede usar BirdNET global.
- Biblioteca de perfiles:
  - Crear/eliminar especies de referencia.
  - Comparacion multi-vector para mayor precision.
- Resultados:
  - Progreso y logs en tiempo real.
  - Resumen con top especies y grafico.
  - Tabla de detecciones con reproduccion por segmento.
- Exportacion en:
  - CSV
  - Raven Table
  - Audacity
  - Kaleidoscope

### Requisitos

- Windows 10/11
- Python **3.11.x** (obligatorio)
- Internet para instalacion inicial

### Inicio rapido

```bat
git clone https://github.com/Guivve-A/Aves_radar.git
cd Aves_radar
start_app.bat
```

`start_app.bat` hace automaticamente:
1. Crea/reutiliza `venv`
2. Configura entorno
3. Ejecuta `test_install.py`
4. Abre la app (`main.py`)

### Instalacion manual

```bat
setup_env.bat
call venv\Scripts\activate
python test_install.py
python main.py
```

### Flujo recomendado de uso

1. Selecciona archivo o carpeta de audio.
2. Elige carpeta de salida.
3. Configura modo de zona (global o con coordenadas).
4. Ajusta preset/parametros.
5. Define formatos de exportacion.
6. Ejecuta analisis.
7. Revisa `Resumen`, `Detecciones`, `Biblioteca de Especies` y `Logs`.

### Datos locales

- `output/`: resultados y exportaciones
- `profiles/`: perfiles locales y metadatos
- `local_models/condesan/`: clasificador local CONDESAN (especies y umbrales calibrados)

### Clasificador local CONDESAN y fine-tuning

- Opcion **Usar clasificador local CONDESAN (BirdNET calibrado)** en Parametros: restringe BirdNET a las 33 especies
  validadas del bosque seco con umbrales calibrados por especie (desactivada por defecto).
- `finetuning/`: pipeline reproducible (datos, embeddings BirdNET, validacion cruzada por sesion, exportacion).
- Pipeline completo en Google Colab sobre la carpeta de Drive (sin descargarla): `finetuning/colab_pipeline.ipynb` ([abrir en Colab](https://colab.research.google.com/github/Guivve-A/Aves_radar/blob/main/finetuning/colab_pipeline.ipynb)). Genera tambien la hoja de anotacion aleatoria para la evaluacion sin sesgo.
- Informe con teoria y resultados: `docs/informe_finetuning/informe_finetuning.pdf`.
- Pestaña **Validación**: carga `TablaFinal.xlsx` y compara cada grabación analizada con las validaciones del equipo (porcentaje de coincidencia, detecciones confirmadas, línea de tiempo por especie). El audio debe estar dentro de la carpeta de su grabadora (`AM001/...`). Para comparar en igualdad de condiciones usa solape 0 s.
- Pruebas: `python -m unittest discover -s tests`.

### Build de ejecutable

```bat
python build_exe.py
```

Para instalador:
1. Instala Inno Setup 6
2. Ejecuta:

```bat
build_installer.bat
```

Salida esperada:
`dist\EcoAcousticSentinel_Installer_x64.exe`

### Problemas comunes

- Python no soportado:
  - Instala Python 3.11.x y vuelve a correr `setup_env.bat`.
- Error en instalacion inicial:
  - Verifica internet y reintenta `setup_env.bat`.
- Sin detecciones:
  - Revisa rutas de audio, formatos soportados, umbrales y modo de zona.
- Ejecutar local no cambia GitHub:
  - GitHub solo cambia con `git add`, `git commit` y `git push`.

## English

### Overview

EcoAcoustic Sentinel combines:
- Local custom species library (few-shot embeddings)
- Global discovery with BirdNET

### Main features

- Guided 4-step workflow:
  1. Audio
  2. Zone
  3. Parameters
  4. Export
- Hybrid strategy:
  - If custom profile hits are found, those results are prioritized.
  - If no custom hit is found, BirdNET global discovery can run.
- Species profile library:
  - Create/remove local reference species.
  - Multi-vector matching for higher precision.
- Results:
  - Real-time progress and logs.
  - Summary with top species and chart.
  - Detections table with segment playback.
- Export formats:
  - CSV
  - Raven Table
  - Audacity
  - Kaleidoscope

### Requirements

- Windows 10/11
- Python **3.11.x** (required)
- Internet connection for first-time setup

### Quick start

```bat
git clone https://github.com/Guivve-A/Aves_radar.git
cd Aves_radar
start_app.bat
```

`start_app.bat` automatically:
1. Creates/reuses `venv`
2. Sets up dependencies
3. Runs `test_install.py`
4. Launches the app (`main.py`)

### Manual setup

```bat
setup_env.bat
call venv\Scripts\activate
python test_install.py
python main.py
```

### Typical workflow

1. Select an audio file or folder.
2. Select output folder.
3. Set zone mode (global or coordinates).
4. Tune preset/parameters.
5. Configure export formats.
6. Start analysis.
7. Review `Summary`, `Detections`, `Species Library`, and `Logs`.

### Local data

- `output/`: analysis outputs and exports
- `profiles/`: local species profiles and metadata
- `local_models/condesan/`: CONDESAN local classifier (species list and calibrated thresholds)

### CONDESAN local classifier and fine-tuning

- **Use CONDESAN local classifier (calibrated BirdNET)** option in Parameters: restricts BirdNET to the 33 validated
  dry-forest species with per-species calibrated thresholds (off by default).
- `finetuning/`: reproducible pipeline (data, BirdNET embeddings, session-grouped cross-validation, export).
- Full pipeline on Google Colab, reading the Drive folder in place: `finetuning/colab_pipeline.ipynb` ([open in Colab](https://colab.research.google.com/github/Guivve-A/Aves_radar/blob/main/finetuning/colab_pipeline.ipynb)). It also builds the random annotation sheet for the unbiased evaluation.
- Report with theory and results (Spanish): `docs/informe_finetuning/informe_finetuning.pdf`.
- **Validación** tab: load `TablaFinal.xlsx` to compare each analysed recording with the team's validations (agreement percentage, confirmed detections, per-species timeline). Keep audio inside its recorder folder (`AM001/...`); use 0 s overlap for a like-for-like comparison.
- Tests: `python -m unittest discover -s tests`.

### Build executable

```bat
python build_exe.py
```

For installer:
1. Install Inno Setup 6
2. Run:

```bat
build_installer.bat
```

Expected output:
`dist\EcoAcousticSentinel_Installer_x64.exe`

### Troubleshooting

- Unsupported Python version:
  - Install Python 3.11.x and rerun `setup_env.bat`.
- First-time setup fails:
  - Check internet access and rerun `setup_env.bat`.
- No detections:
  - Verify audio paths, supported formats, thresholds, and zone mode.
- Local execution does not change GitHub:
  - GitHub changes only after `git add`, `git commit`, and `git push`.

## Scientific Credit

BirdNET (Kahl et al., 2021):
BirdNET: A deep learning solution for avian diversity monitoring.
Ecological Informatics, 61, 101236.

## License

MIT. See `LICENSE`.
