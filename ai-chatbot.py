import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # Alternatively, use "gpt2-medium" for better quality if you have sufficient resources
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate a response
def generate_response(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to handle sending a message
def send_message():
    user_input = user_entry.get()
    if user_input.lower() == 'exit':
        window.destroy()
        return

    # Display user's message in chat window
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "You: " + user_input + "\n")
    chat_box.config(state=tk.DISABLED)
    
    # Clear entry field
    user_entry.delete(0, tk.END)

    # Generate and display chatbot response
    prompt = f"User: {user_input}\nAI:"
    response = generate_response(prompt)
    chat_box.config(state=tk.NORMAL)
    chat_box.insert(tk.END, "Chatbot: " + response + "\n\n")
    chat_box.config(state=tk.DISABLED)

# Set up GUI window
window = tk.Tk()
window.title("AI Chatbot")
window.geometry("600x400")
window.resizable(width=False, height=False)

# Chat display area
chat_box = scrolledtext.ScrolledText(window, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 12))
chat_box.place(x=10, y=10, width=580, height=300)

# User input field
user_entry = tk.Entry(window, font=("Arial", 14))
user_entry.place(x=10, y=320, width=480, height=40)

# Send button
send_button = tk.Button(window, text="Send", command=send_message, font=("Arial", 14))
send_button.place(x=500, y=320, width=80, height=40)

# Instructions
instructions = tk.Label(window, text="Type 'exit' to close the chatbot.", font=("Arial", 10), fg="gray")
instructions.place(x=10, y=370)

# Start the GUI loop
window.mainloop()
