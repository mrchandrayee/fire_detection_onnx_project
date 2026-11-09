from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import cv2
from detection.ort_detector import OrtDetector

def load_names(path):
    if not path: return None
    p=Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Class names file not found: {p}")
    if p.suffix.lower()==".json":
        data=json.loads(p.read_text())
        if isinstance(data, list): return data
        return [data[str(i)] if str(i) in data else data[i] for i in range(len(data))]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]

def main():
    ap=argparse.ArgumentParser(description="Fire/smoke detection from video using YOLO ONNX + ONNX Runtime")
    ap.add_argument("--source",default="input.mp4",help="video path or webcam index (0)")
    ap.add_argument("--model",default="models/fire_smoke.onnx",help="trained fire/smoke ONNX model")
    ap.add_argument("--names",default="classes.txt",help="class names in model order")
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--conf",type=float,default=0.35)
    ap.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    ap.add_argument("--output",default="fire_detection_output.mp4")
    ap.add_argument("--show",action="store_true")
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"Cannot open video source: {args.source}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    detector=OrtDetector(args.model,args.device,args.imgsz,args.conf,load_names(args.names))
    palette={"fire":(0,0,255),"smoke":(128,128,128)}
    total_frames=0; positive_frames=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        detections=detector.predict(frame)
        total_frames+=1
        if detections: positive_frames+=1
        for det in detections:
            x1,y1,x2,y2=map(int,det.xyxy)
            color=palette.get(det.class_name.lower(),(0,180,255))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label=f"{det.class_name} {det.confidence:.2f}"
            top=max(0,y1-24)
            cv2.rectangle(frame,(x1,top),(min(width-1,x1+max(100,len(label)*9)),y1),color,-1)
            cv2.putText(frame,label,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        status="FIRE/SMOKE DETECTED" if detections else "No detection"
        color=(0,0,255) if detections else (0,200,0)
        cv2.putText(frame,status,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
        writer.write(frame)
        if args.show:
            cv2.imshow("Fire and smoke detection",frame)
            if cv2.waitKey(1)&0xFF==ord("q"): break
    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Saved annotated video: {args.output}")
    print(f"Frames with detections: {positive_frames}/{total_frames}")

if __name__=="__main__": main()
