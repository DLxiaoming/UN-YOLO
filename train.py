import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'UN-YOLO\ultralytics-main\ultralytics\cfg\models\v8\yolov8s-c2f-dwr-sli-dys.yaml') # select yaml file
    model.load(r'UN-YOLO\ultralytics-main\weights\yolov8s.pt') # loading pretrain weights
    model.train(data=r'UN-YOLO\ultralytics-main\XXX-DET\data.yaml', # select data set 
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                patience=0,
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )