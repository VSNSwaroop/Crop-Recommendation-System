# Crop Recommendation System

An open-source machine-learning web application that combines **soil-image classification** with **temperature and rainfall data** to recommend a suitable crop.

The project contains two ML stages:

1. A convolutional neural network (CNN) classifies the uploaded soil image.
2. A crop classifier combines the detected soil type with temperature and rainfall to recommend a crop.

The application is served through Flask and includes account registration, administrator activation, and a farmer prediction dashboard.

## Features

- Soil image classification with a TensorFlow/Keras CNN
- Crop recommendation using soil type, temperature, and rainfall
- Optional browser geolocation + Open-Meteo weather lookup
- Flask authentication and user activation workflow
- SQLite for local development
- CSRF protection, password hashing, upload limits, and security headers
- Training scripts for both ML stages
- GitHub Actions CI for lightweight quality and security checks

## Current scope

The crop model currently uses the soil categories and crop examples provided in `crop_data.csv`. The soil CNN uses the image folders under `dataset/Train` and `dataset/test`.

This project is intended for **learning, experimentation, and open-source development**. Its recommendations should not be treated as professional agronomic advice or as a replacement for soil testing, local agricultural guidance, or field-specific analysis.

## Architecture

```text
Soil image
   |
   v
Soil CNN --------------------+
                             |
                             v
                       Detected soil type
                             |
Temperature + Rainfall ------+
                             |
                             v
                       Crop classifier
                             |
                             v
                     Recommended crop
```

## Repository structure

```text
.
├── app.py                    # Flask application and prediction endpoint
├── crop_data.csv             # Tabular crop training data
├── crop_model.h5             # Trained crop model
├── soil_cnn_model.h5         # Trained soil-image CNN
├── crop_encoder.pkl          # Crop label encoder
├── soil_encoder.pkl          # Soil feature encoder
├── train_crop_model.py       # Crop model training script
├── train_soil_cnn.py         # Soil CNN training script
├── dataset/
│   ├── Train/
│   └── test/
├── templates/                # Flask/Jinja user interface
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/VSNSwaroop/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development tools:

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and set your own values.

At minimum, set a strong `SECRET_KEY`. To create the first admin account, also set `ADMIN_EMAIL` and `ADMIN_PASSWORD`.

Example:

```env
SECRET_KEY=replace-with-a-long-random-secret
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=replace-with-a-strong-password
```

Never commit your real `.env` file.

### 5. Run the application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Training the models

### Crop model

```bash
python train_crop_model.py
```

This regenerates:

- `crop_model.h5`
- `soil_encoder.pkl`
- `crop_encoder.pkl`

### Soil CNN

```bash
python train_soil_cnn.py
```

This regenerates:

- `soil_cnn_model.h5`
- `soil_labels.json`

The generated `soil_labels.json` preserves the class-to-output order learned by Keras so inference does not rely on an assumed directory ordering.

## Security

Do not put secrets, real user databases, API keys, or passwords in the repository.

Security reports should follow [SECURITY.md](SECURITY.md).

## Testing

Run lightweight tests with:

```bash
pytest -q
```

Run static checks with:

```bash
ruff check .
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or pull request.

Useful contribution areas include:

- model evaluation and reproducibility
- better datasets and dataset documentation
- accessibility and UI improvements
- automated tests
- security hardening
- deployment documentation
- explainability and confidence calibration

## Roadmap

- [ ] Add model evaluation reports and confusion matrices
- [ ] Expand and document the crop dataset
- [ ] Add stronger automated tests for model inference
- [ ] Add database migrations
- [ ] Add rate limiting for authentication and prediction endpoints
- [ ] Improve model explainability
- [ ] Package model metadata and version information
- [ ] Publish versioned releases

## Maintainer

Primary maintainer: [VSNSwaroop](https://github.com/VSNSwaroop)

## License

Licensed under the [MIT License](LICENSE).
