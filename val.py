import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'ultralytics-20240222\runs\exp\weights\best.pt') 
    model.val(data=r'ultralytics-20240222\ultralytics-main\POT-DET\data.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )