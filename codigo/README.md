# Filtrado de ruidos no estacionarios con redes neuronales y filtros adaptativos

## Generación de base de datos

1. Colocar en el directorio raw-dataset los directorios clean_test, clean_train, noise_test y noise_train, tomados de https://arxiv.org/abs/1909.08050
2. Correr los siguientes comandos
    * datasets.py generate --env=test
    * datasets.py generate --env=train


## Entrenar red

1. Execute train script 

```
python -m dnntrain --epochs=10 --input-dir=./dataset/audios_train --output-dir=./trained-models --experiment-name=dnn-train
```
2. Execute tensorboard

```
tensorboard --logdir=./trained-models/logs
```