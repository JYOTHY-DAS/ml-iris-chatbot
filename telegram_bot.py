import requests
import telebot

TOKEN = "7855313200:AAGyj6DxPnGpwa2qrFSqyLwE7o_8YGxSbXI"
API_URL = "http://127.0.0.1:8000/predict/"

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Send me iris flower measurements in this format: \n4.9 3.1 1.5 0.2")

@bot.message_handler(func=lambda msg: True)
def predict_iris_species(message):
    try:
        values = list(map(float, message.text.split()))
        if len(values) != 4:
            raise ValueError("Incorrect number of features.")
        
        response = requests.post(API_URL, json={
            "sepal_length": values[0],
            "sepal_width": values[1],
            "petal_length": values[2],
            "petal_width": values[3]
        })
        
        result = response.json()
        bot.reply_to(message, f"Predicted Species: {result['species']}")
    except Exception as e:
        bot.reply_to(message, f"Error: {str(e)}")

bot.polling()
