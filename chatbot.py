from dotenv import load_dotenv
from agent import SupportAgent
import os

load_dotenv()

chatbot = SupportAgent()
exit = False

while not exit:
    message = input("Enter your query (E to exit): ")
    
    if message.upper() == "E":
        exit = True

    else:
        print(f"\nAgent:\n{chatbot.chat(message)}")

print("Thank you for visiting Vela!")