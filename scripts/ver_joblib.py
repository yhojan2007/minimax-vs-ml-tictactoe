import joblib
from sklearn.tree import export_text
from scripts.generar_dataset import FEATURE_COLUMNS

# 1. Cargar el archivo .joblib
model_path = "models/tree_model.joblib"
data = joblib.load(model_path)

print("=== CONTENIDO DEL ARCHIVO JOBLIB ===")
print(f"Llaves disponibles: list(data.keys())")
print(f"Precisión guardada del modelo: {data['accuracy']:.1%}\n")

# 2. Extraer el objeto del Árbol de Decisión (DecisionTreeClassifier)
tree = data["tree"]

print("=== PROPIEDADES DEL ÁRBOL ===")
print(f"Profundidad máxima: {tree.get_depth()}")
print(f"Número de nodos: {tree.tree_.node_count}")
print(f"Número de hojas: {tree.get_n_leaves()}")
print(f"Clases de salida (jugadas posibles): {tree.classes_}\n")

print("=== REGLAS APRENDIDAS (Primeros niveles) ===")
# Muestra las reglas lógicas que el árbol usa para decidir
rules = export_text(tree, feature_names=FEATURE_COLUMNS, max_depth=4)
print(rules)