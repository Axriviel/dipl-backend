# Použij základní image s Pythonem
FROM python:3.12.4

# Nastav pracovní adresář
WORKDIR /app

# Zkopíruj soubor requirements.txt a nainstaluj závislosti
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Zkopíruj zbytek aplikace do pracovního adresáře
COPY . .

# Exponuj port, na kterém Flask běží (standardně 5000)
EXPOSE 5000

RUN ls -la /app

# Definuj příkaz pro spuštění aplikace
CMD ["python", "app.py"]
