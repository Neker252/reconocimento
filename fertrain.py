import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Configuración
num_features = 64
num_labels = 7
batch_size = 64
epochs = 30   # Más épocas porque early stopping controlará
width, height = 48, 48

train_dir = 'train'
test_dir = 'test'

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

X = []
Y = []

print("⏳ Cargando imágenes...")

for idx, emotion in enumerate(emotions):
    folder_path = os.path.join(train_dir, emotion)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ No existe la carpeta: {folder_path}")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))
            X.append(img)
            Y.append(idx)
        except:
            continue

# Convertir a numpy
x = np.array(X, dtype='float32') / 255.0
y = np_utils.to_categorical(Y, num_classes=num_labels)

# ========== REDUCIR DATOS PARA PRUEBA RÁPIDA ==========
print(f"\n📊 Imágenes originales cargadas: {len(x)}")

# Toma solo el 20% para pruebas rápidas
subset_size = len(x) // 5  # 20%
x = x[:subset_size]
y = y[:subset_size]

print(f"🚀 Usando {len(x)} imágenes para prueba rápida (20% del total)")
print(f"✅ Esto reducirá el tiempo de entrenamiento aprox. 80%")
# ========== FIN REDUCCIÓN ==========

# Normalización
x -= np.mean(x)
x /= (np.std(x) + 1e-7)

# Formato para CNN (N, 48, 48, 1)
x = x.reshape(-1, width, height, 1)

print("✅ Datos cargados:")
print("x:", x.shape)
print("y:", y.shape)

# Dividir en sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

print(f"📊 Divisiones:")
print(f"  Entrenamiento: {len(X_train)} imágenes")
print(f"  Validación: {len(X_valid)} imágenes")
print(f"  Prueba: {len(X_test)} imágenes")

# Guardar muestras de test
np.save('modXtest.npy', X_test)
np.save('modytest.npy', y_test)

# ========== MODELO ==========
model = Sequential()

model.add(Conv2D(num_features, (3, 3), activation='relu', input_shape=(width, height, 1), kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(4*num_features, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(4*num_features, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))

# ========== COMPILAR CON RMSprop ==========
model.compile(
    loss=categorical_crossentropy,
    optimizer=RMSprop(lr=0.0005),  # Optimizador para imágenes
    metrics=['accuracy']
)

print("\n" + "="*60)
print("CONFIGURACIÓN DEL MODELO:")
print("="*60)
print(f"• Optimizador: RMSprop (lr=0.0005)")
print(f"• Épocas máximas: {epochs}")
print(f"• Batch size: {batch_size}")
print(f"• Neuronas Dense: 256 → 128 → 64")
print(f"• Activaciones: relu → tanh → relu")
print("="*60 + "\n")

print("✅ Configurando Early Stopping (para en 5 épocas sin mejorar)")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=9,
    restore_best_weights=True,
    verbose=1
)

print("🚀 Entrenando modelo con AUMENTO DE DATOS y EARLY STOPPING...")

# ========== AUMENTO DE DATOS ==========
datagen = ImageDataGenerator(
    rotation_range=15,  
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Calcular steps por época
steps_per_epoch = len(X_train) // batch_size
print(f"📊 Steps por época: {steps_per_epoch}")

# Entrenar con fit_generator (más compatible)
history = model.fit_generator(
    generator=datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

# Guardar modelo
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)

model.save_weights("fer.h5")

print("\n" + "="*60)
print("RESUMEN DEL ENTRENAMIENTO:")
print("="*60)
print(f"✅ Modelo guardado: fer.json y fer.h5")
print(f"• Épocas ejecutadas: {len(history.history['loss'])} de {epochs}")
if 'val_acc' in history.history:
    print(f"• Mejor precisión en validación: {max(history.history['val_acc'])*100:.2f}%")
if 'val_loss' in history.history:
    print(f"• Pérdida final en validación: {history.history['val_loss'][-1]:.4f}")
print("="*60)

# Probar inmediatamente
print("\n🎯 Probando modelo con datos de test...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Precisión en test: {test_acc*100:.2f}%")
print(f"✅ Pérdida en test: {test_loss:.4f}")