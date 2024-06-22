import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Muat model yang sudah dilatih
model = tf.keras.models.load_model('VGG16.h5')

# Daftar nama kelas
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams']  # Ganti dengan nama kelas yang sesuai dengan model Anda

# Fungsi untuk memuat dan memproses gambar
def load_image(filename):
    img = Image.open(filename)
    img = img.resize((224, 224))  # Ubah ukuran sesuai dengan ukuran input model
    img = np.array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    return img

# Fungsi untuk klasifikasi gambar
def classify_image():
    global img_path
    if img_path:
        progress.start()
        img = load_image(img_path)
        predictions = model.predict(img)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_idx]
        confidence = np.max(predictions)
        result_text.set(f"Predicted Class: {predicted_class_name}, Confidence: {confidence:.2f}")
        progress.stop()
    else:
        messagebox.showwarning("Warning", "Please select an image first!")

# Fungsi untuk memilih gambar
def select_image():
    global img_path
    img_path = filedialog.askopenfilename()
    if img_path:
        load_img = Image.open(img_path)
        load_img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(load_img)
        panel.configure(image=img)
        panel.image = img
        result_text.set("")

# Setup GUI menggunakan Tkinter
root = tk.Tk()
root.title("Image Classification")
root.geometry("500x500")
root.configure(bg='#F0F0F0')

img_path = ""

# Frame untuk menampilkan gambar
frame_image = tk.Frame(root, bg='#F0F0F0')
frame_image.pack(side="top", pady=20)

# Label untuk menampilkan gambar yang dipilih
panel = tk.Label(frame_image, bg='#F0F0F0')
panel.pack()

# Frame untuk tombol
frame_buttons = tk.Frame(root, bg='#F0F0F0')
frame_buttons.pack(side="top", pady=20)

# Tombol untuk memilih gambar
btn_select = tk.Button(frame_buttons, text="Select Image", command=select_image, bg='#4CAF50', fg='white', padx=10, pady=5)
btn_select.grid(row=0, column=0, padx=10)

# Tombol untuk mengklasifikasikan gambar
btn_classify = tk.Button(frame_buttons, text="Classify Image", command=classify_image, bg='#2196F3', fg='white', padx=10, pady=5)
btn_classify.grid(row=0, column=1, padx=10)

# Progress Bar
progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
progress.pack(pady=20)

# Frame untuk hasil
frame_result = tk.Frame(root, bg='#F0F0F0')
frame_result.pack(side="bottom", pady=20)

# Label untuk menampilkan hasil klasifikasi
result_text = tk.StringVar()
result_label = tk.Label(frame_result, textvariable=result_text, font=('Helvetica', 16), bg='#F0F0F0')
result_label.pack()

root.mainloop()
