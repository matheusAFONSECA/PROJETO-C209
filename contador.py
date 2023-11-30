# CÓDIGO QUE REALIZA A CONTAGEM DE LARANJAS APÓS PASSAR POR UMA LINHA

# importação de bibliotecas
import cv2
from ultralytics import YOLO


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

            cv2.imshow('Video', frame)      # mostra a imagem dos frames (vídeo da webcam)

            # Verificar se a tecla 'q' foi pressionada para encerrar o programa
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if success:
                # Roda a inferência do YoloV8 no frame fazendo uma predição
                results = model.predict(frame, verbose=False, show=True, conf=0.5)

    except Exception as ex:
        print(ex)


try:
    print("Iniciando Detecção de laranjas...")
    while 1:
        detect()                 # chama a função de detecção

except Exception as e:       # caso aconteça algum erro é printado o mesmo
    print(e)

