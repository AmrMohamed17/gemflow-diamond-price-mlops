import os
import pickle
from sklearn.metrics import r2_score


def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj)

  except Exception as e:
    print(e)


def evaluate_models(X_train, X_test, y_train, y_test,models):
  try:
    report = {}

    for model_name, model in models.items():
      print(f"started {model_name}")
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      r2 = r2_score(y_test, y_pred)
      report[model_name] = r2
      print(f"{model_name} Ended Successfully")

    return report
  except Exception as e:
    print(e)


def load_object(file_path):
  try:
    with open(file_path, "rb") as file_obj:
      return pickle.load(file_obj)
  except Exception as e:
    print(e)