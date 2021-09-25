import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util, visualization_utils

# Configurations des répertoires
WORKSPACE_PATH = 'boy_and_girl/workspace'
SCRIPTS_PATH = 'boy_and_girl/scripts'
APIMODEL_PATH = 'boy_and_girl/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

# Configuration du pipeline de configuration
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

# Construction du model de detection
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Chargement d'un checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Configuration de la capture vidéo
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    # Configurations des détections : noms, numéros, classes
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Visualisation objet détecté avec libellé
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False)

    # Affichage de la fenêtre fin prête
    cv2.imshow('Know The Gender App - Boy And Girl Real Time detection', cv2.resize(image_np_with_detections, (800, 600)))

    # Appuyer sur la touche "q" arrêtera l'application immiédiatement
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
