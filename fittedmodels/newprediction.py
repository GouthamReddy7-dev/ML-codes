import joblib
model=joblib.load('weather_model.pkl')
new_data=[[0.5,24,15,10]]
pred=model.predict(new_data)
if 1 in pred:
  print("Rain predicted Today !")
elif 2 in pred:
  print("Sun predicted Today !")
elif 3 in pred:
  print("Fog predicted Today !")
elif 4 in pred:
  print("Dizzle predicted Today !")
else:
  print("Sorry couldn't predict correctly")
print(pred)