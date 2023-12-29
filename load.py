import tensorflow as tf

# ID du fichier sur Google Drive
file_id = '1bS3R3rSyWLJswJI0M_qD1L1y-UGkNnjU'

# URL de l'API Google Drive pour télécharger le fichier
url = f'https://drive.google.com/uc?id={file_id}'

# Utilisez la méthode `tf.keras.utils.get_file` pour charger le modèle directement depuis Google Drive
model_path = tf.keras.utils.get_file('modele_french.h5', origin=url, extract=False, cache_subdir='/content/drive/MyDrive/model/')

# Chargez le modèle
model = tf.keras.models.load_model(model_path)

# Faites une prédiction
# prediction = model.predict(data)
