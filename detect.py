import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'UN-YOLO\runs\exp\weights\best.pt') 
    model.predict(source=r'UN-YOLO\ultralytics-main\POT-DET\images\val',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # save_txt=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )