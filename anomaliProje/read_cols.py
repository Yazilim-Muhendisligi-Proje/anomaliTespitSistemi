import joblib
import pandas as pd
import json

cols = joblib.load('c:\\Users\\Mehmet Ersolak\\Desktop\\Yazılım Mühendisliği projesi\\anomaliTespitSistemi\\anomaliProje\\model_columns.pkl')
print("COLUMNS_START")
print(json.dumps(list(cols)))
print("COLUMNS_END")
