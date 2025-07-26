# AI Crop Predictor Web App

This is a simple AI-powered web app that:
- Accepts a soil image
- Accepts weather data (temperature + rainfall)
- Uses a placeholder (or your own) LSTM-CNN model to predict the most suitable crop
- Displays the result on a clean web page

## Features
- Upload a soil image
- Enter temperature and rainfall
- Get a crop prediction (currently a dummy result; replace with your model logic)

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   python app.py
   ```
4. **Open your browser** and go to [http://localhost:5000](http://localhost:5000)

## File Structure
- `app.py` - Flask backend
- `templates/index.html` - Frontend HTML
- `uploads/` - Temporary image storage (auto-created)

## Deployment
You can deploy this app to platforms like **Render**, **Heroku**, or **Railway**.

### Example: Deploy to Render
1. Push your code to a public GitHub repository.
2. Go to [Render.com](https://render.com/), create a new Web Service, and connect your repo.
3. Set the build and start commands:
   - Build: `pip install -r requirements.txt`
   - Start: `python app.py`
4. Add a web service environment variable if needed: `PORT=10000` (and update `app.py` to use `os.environ.get('PORT', 5000)`)

## Custom Model Integration
- Replace the dummy crop prediction logic in `app.py` with your own LSTM-CNN model inference code.
- Make sure to handle image preprocessing and weather data as required by your model.

## License
MIT "# crop-recommendation-system" 
