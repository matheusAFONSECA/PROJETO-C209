# CÓDIGO QUE REALIZA A CONTAGEM DE LARANJAS EM UMA ÁREA ESPECÍFICA

# importação de bibliotecas
import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv

# -> definindo o poligono:
# site para pegar as coordenadas do poligono: https://roboflow.github.io/polygonzone/
# cor do poligono que ira aparecer na imagem
colors = sv.ColorPalette.default()

# coordenadas do poligono
polygons = [np.array([[60, 110], [228, 106], [228, 302], [48, 306]])]


def detect():       # função que realiza a detecção de laranjas no video
    try:
        # Abre o vídeo de captura das imagens -> 0: webcan do notebook, 1: porta USB...
        cap = cv2.VideoCapture(0)

        # definido o modelo de rede que foi treinado para detectar laranjas
        model = YOLO("runs\\detect\\train\\weights\\best.pt")

        flag = True     # flag de controle

        while flag:

            # Lê os frames do vídeo capturados pela câmera
            success, frame = cap.read()

            if not success:
                break

            frame_resolution_wh = (frame.shape[1], frame.shape[0])
            zones = [sv.PolygonZone(polygon=polygon, frame_resolution_wh=frame_resolution_wh) for polygon in polygons]
            zone_annotators = [
                sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=4, text_thickness=8,
                                        text_scale=4) for index, zone in enumerate(zones)]
            box_annotators = [sv.BoxAnnotator(color=colors.by_idx(index), thickness=4, text_thickness=4, text_scale=2)
                              for index in range(len(polygons))]

            results = model(frame, imgsz=224)[0]
            detections = sv.Detections.from_yolov8(results)

            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                mask = zone.trigger(detections=detections)
                detections_filtered = detections[mask]
                frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
                frame = zone_annotator.annotate(scene=frame)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as ex:
        print(ex)


try:
    print("Iniciando Detecção de laranjas...")
    while 1:
        detect()                 # chama a função de detecção

except Exception as e:       # caso aconteça algum erro é printado o mesmo
    print(e)

