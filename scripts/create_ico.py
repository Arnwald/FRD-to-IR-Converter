from PIL import Image
import os

# Chemin vers ton logo PNG haute résolution (ex: 1024x1024)
PNG_PATH = "../assets/icon.png"
ICO_PATH = "../icons/icon.ico"

# Les tailles standard pour Windows
sizes = [(16,16), (32,32), (48,48), (256,256)]

im = Image.open(PNG_PATH)

# Convertir en carré RGBA si nécessaire
im = im.convert("RGBA")

# Générer .ico multi-résolution
im.save(ICO_PATH, format='ICO', sizes=sizes)
print(f"Saved {ICO_PATH}")
