import cv2
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from ultralytics import YOLO
import numpy as np

# Função para obter a cor média de uma região de interesse
def get_average_color(roi):
    avg_color_per_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

# Função para converter RGB para HSV
def rgb_to_hsv(rgb_color):
    rgb_color = np.uint8([[rgb_color]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2HSV)[0][0]
    return hsv_color

# Função para identificar a cor na região de interesse
def identify_color(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]
    
    avg_color = get_average_color(roi)
    avg_hsv_color = rgb_to_hsv(avg_color)
    
    hue = avg_hsv_color[0]
    saturation = avg_hsv_color[1]
    value = avg_hsv_color[2]
    
    # Lógica para determinar o nome da cor
    if value < 25:
        return "Preto"
    elif saturation < 38 and value > 200:
        return "Branco"
    elif saturation < 25 and 26 < value < 185:
        return "Cinza"
    elif hue < 5:
        return "Vermelho"
    elif hue < 22:
        return "Laranja"
    elif hue < 33:
        return "Amarelo"
    elif hue < 78:
        return "Verde"
    elif hue < 131:
        return "Ciano"
    elif hue < 167:
        return "Azul"
    elif hue < 260:
        return "Roxo"
    else:
        return "Desconhecido"
    
class YOLOApp(App):
    def build(self):
        self.model = YOLO('modeloTreinado.pt')  # Carregar o modelo treinado YOLO
        self.capture = cv2.VideoCapture(0)  # Capturar vídeo da webcam
        
        # Layout da aplicação
        self.layout = BoxLayout(orientation='vertical')
        self.img = Image()
        self.layout.add_widget(self.img)
        
        self.button_layout = BoxLayout(size_hint_y=None, height='48dp')
        self.start_button = Button(text='Iniciar Deteção')
        self.stop_button = Button(text='Parar Deteção', disabled=True)
        self.start_button.bind(on_press=self.start_detection)
        self.stop_button.bind(on_press=self.stop_detection)
        self.button_layout.add_widget(self.start_button)
        self.button_layout.add_widget(self.stop_button)
        
        self.layout.add_widget(self.button_layout)
        return self.layout

    def start_detection(self, instance):
        self.start_button.disabled = True
        self.stop_button.disabled = False
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def stop_detection(self, instance):
        self.start_button.disabled = False
        self.stop_button.disabled = True
        Clock.unschedule(self.update)
        self.img.texture = None

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            results = self.model(frame)
            for result in results[0].boxes:
                bbox = result.xyxy[0]
                conf = result.conf[0]
                cls = result.cls[0]
                x1, y1, x2, y2 = map(int, bbox)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                color_name = identify_color(frame, (x1, y1, x2, y2))

                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                color_text = f"Cor: {color_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, color_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    YOLOApp().run()
