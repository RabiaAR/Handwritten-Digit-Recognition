import tkinter as tk
from tkinter import Canvas
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.model = tf.keras.models.load_model('model.h5')

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict_digit(self):
        self.canvas.update()
        ps = self.canvas.postscript(colormode='mono')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img = img.resize((28, 28))
        img = ImageOps.grayscale(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        prediction = self.model.predict(img)
        digit = np.argmax(prediction)
        tk.messagebox.showinfo("Prediction", f"The predicted digit is: {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
