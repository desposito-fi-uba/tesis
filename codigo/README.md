# Filtrado de ruidos no estacionarios con redes neuronales y filtros adaptativos

## Generación de base de datos

1. Colocar en el directorio raw-dataset los directorios clean_test, clean_train, noise_test y noise_train, tomados de https://arxiv.org/abs/1909.08050
2. Correr los siguientes comandos
    * `python -m datasets generate --env=test`
    * `python -m datasets generate --env=train`

## Generación de base de datos con un solo tipo de ruido por audio

 * `python -m datasets generate --env=test --use-one-noise-type-per-audio=1 --output-dir=dataset_noise_type`

## Niveles base de PESQ y STOI

1. Generar los valores con: `python -m datasets compute-pesq-stoi-histogram  --input-dir=./dataset`
2. Graficar con: `python -m datasets plot-pesq-stoi-histogram   --input-dir=./dataset`

## Niveles base de PESQ y STOI con tipo de ruido

1. Generar los valores con: `python -m datasets compute-pesq-stoi-histogram --input-dir=./dataset_with_noise_type`
2. Graficar con: `python -m datasets plot-pesq-stoi-histogram --input-dir=./dataset_with_noise_type`

## Estadísticas de los conjuntos

1. Conjunto de entrenamiento `python -m datasets statistics --env=train`
2. Conjunto de pruebas `python -m datasets statistics --env=test`


## Entrenar red localmente

1. Ejecutar script de entrenamiento 

```
python -m dnntrain --epochs=10 --input-dir=./dataset --output-dir=./trained-models --experiment-name=dnn-train
```

2. Ejecutar tensorboard

```
tensorboard --logdir=./trained-models/logs
```

## Entrenar red localmente usando resources de gcp

1. Ejecutar entrenamiento 

```
python -u -m dnntrain --epochs=10 --input-dir=gs://desposito-noisefilter/dataset_reduced.tar.gz --output-dir=gs://desposito-noisefilter --experiment-name=dnn-train --overload-settings=gs://desposito-noisefilter/overload_settings.json
```

2. Ejecutar tensorboard

```
tensorboard --logdir=gs://desposito-noisefilter/logs
```

## Entrenar red en gcp

### Configurar gcp

Seguir pasos en: https://cloud.google.com/ai-platform/training/docs/using-containers

### Subir resources

#### Dataset

* Comprimir `tar -czvf dataset.tar.gz ./dataset`
* Subir `gsutil cp ./dataset.tar.gz gs://desposito-noisefilter`

#### Configuración de entrenamiento

* Subir `gsutil cp ./overload_settings.json gs://desposito-noisefilter`

### Imagen de docker

#### Creación

* Obtener id de proyecto creado en el paso anterior usando `gcloud config list project --format "value(core.project)"`
* Exportar variables de entorno

```
export PROJECT_ID=tesis-372723
export GCLOUD_PROJECT=tesis-372723
export IMAGE_REPO_NAME=trainer-noise-filter
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
```

* Ejecutar `docker build -f ./trainer.dockerfile -t $IMAGE_URI ./`

#### Correr la imagen en modo interactivo

* Ejecutar `docker run -it -e GCLOUD_PROJECT -v ~/.config/gcloud:/root/.config/gcloud $IMAGE_URI bash`

#### Probar entrenamiento localmente

* Exportar variables

```
export PROJECT_ID=tesis-372723
export GCLOUD_PROJECT=tesis-372723
export IMAGE_REPO_NAME=trainer-noise-filter
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
```

* Ejecutar `docker run -v ~/.config/gcloud:/root/.config/gcloud -e GCLOUD_PROJECT --entrypoint "bash" $IMAGE_URI -c 'cd /root/code && python -u -m dnntrain --epochs=10 --input-dir=gs://desposito-noisefilter/dataset.tar.gz --output-dir=gs://desposito-noisefilter --experiment-name=dnn-train --overload-settings=gs://desposito-noisefilter/overload_settings.json'`

#### Subirla al registry de gcp

* Subir la imagen `docker push $IMAGE_URI`

### Entrenar

```
export PROJECT_ID=tesis-372723
export GCLOUD_PROJECT=tesis-372723
export IMAGE_REPO_NAME=trainer-noise-filter
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG
export now=$(date +"%Y%m%d_%H%M%S")
export JOB_NAME="noise_filter_training_$now"

gcloud ai-platform jobs submit training $JOB_NAME \
  --region us-west1 \
  --master-image-uri $IMAGE_URI \
  --config=aiplatformconfig.yaml
```

### Visualizar entrenamiento

* Ejecutar `tensorboard --logdir=gs://desposito-noisefilter/logs`

## Probar el modelo de dnn

1. Ejecutar script de pruebas 

```
python -m dnnpredict --input-dir=./dataset --output-dir=./trained-models --experiment-name=dnn-test
```

2. Ejecutar tensorboard

```
tensorboard --logdir=./trained-models/logs
```

## Probar el modelo de dnn contra la db que contiene tipos de ruidos

```
python -m dnnpredict --input-dir=./dataset_with_noise_type --output-dir=./trained-models --experiment-name=dnn-test-noise-type
```

## Probar el filtro adaptativo

```
python -m afpredict --input-dir=./dataset
```

## Probar el filtro adaptativo contra la db que contiene tipos de ruidos

```
python -m afpredict --input-dir=./dataset_with_noise_type
```

## Obtener métricas PESQ y STOI para filtro neuronal

```
python -m plotter plot-pesq-stoi --input-dir=./dataset  --filter-type=dnn
```

## Obtener métricas PESQ y STOI por snr y tipos de ruido para filtro neuronal

```
python -m plotter plot-pesq-stoi --input-dir=./dataset_with_noise_type --filter-type=dnn
```

## Obtener métricas PESQ y STOI para filtro adaptativo

```
python -m plotter plot-pesq-stoi --input-dir=./dataset  --filter-type=af
```

## Obtener métricas PESQ y STOI por snr y tipos de ruido para filtro adaptativo

```
python -m plotter plot-pesq-stoi --input-dir=./dataset_with_noise_type  --filter-type=af
```

## Obtener métricas de entrenamiento del filtro neuronal

```
python -m analyzelogs analyze-train-data --output-dir=./trained-models/logs --experiment-names="dnn-train-time-decay1e-4-lr1e-3-w60-batch400-3layersnet, dnn-train-time-decay1e-3-lr1e-4-w60-batch500-3layersnet"
```

## Obtener nivel de ruido y ecm para el filtro neuronal

```
python -m plotter plot-ecm-and-noise-level --input-dir=./dataset  --filter-type=dnn
```

## Obtener nivel de ruido y ecm para el filtro adaptativo

```
python -m plotter plot-ecm-and-noise-level --input-dir=./dataset  --filter-type=af
```

## Comparar pesq y stoi entre filtro adaptativo y filtro neuronal

```
python -m plotter compare-pesq-stoi-results
```