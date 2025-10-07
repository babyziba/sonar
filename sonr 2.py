# Let the OS run this file using whatever `python3` it finds first in PATH
#!/usr/bin/env python3
# A short description of what this script does and how to run it
"""
YOLOv11 Webcam Demo (minimal)
- Opens a webcam (or video file) and runs YOLOv11.
- Draws bounding boxes + labels. Press 'q' to quit.

Usage:
  python app.py                 # webcam index 0
  python app.py --source 1      # webcam index 1
  python app.py --source video.mp4
  python app.py --model yolo11n.pt
"""
#Standard library helper to parse --model and --source from the command line
import argparse
#OpenCV: used to grab frames from the camera/video and to show a window
import cv2
#Ultralytics’ YOLO class: wraps model loading + inference
from ultralytics import YOLO

#Small helper to open either a video file or a webcam cleanly on macOS
def open_capture(src_arg: str):
    #Docstring explaining what this helper expects and returns
    """Open webcam if src_arg is a digit, else open as file."""
    #If the user passed a non-digit (e.g., "video.mp4"), treat it as a file path
    #and try to open it directly
    if src_arg and not src_arg.isdigit():
        cap = cv2.VideoCapture(src_arg)
        #If OpenCV managed to open the file, we’re good—return the capture handle
        if cap.isOpened():
            return cap
        #If not, bail with a clear error message so the user knows what went wrong
        raise SystemExit(f"Could not open video file: {src_arg}")

    #At this point we expect a webcam index (like "0" or "1")
    # onvert to int, default to 0 if the string was empty
    idx = int(src_arg) if src_arg else 0
    #On macOS, AVFoundation is the most reliable backend; if that fails, try CAP_ANY
    for api in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        #Ask OpenCV to open the webcam at the given index with the chosen backend
        cap = cv2.VideoCapture(idx, api)
        #If the device opened, double-check that it can actually deliver a frame
        if cap.isOpened():
            ok, _ = cap.read()
            #Only accept the device if reading a frame works; otherwise release and try next
            if ok:
                return cap
            cap.release()
    #If we tried the likely backends and still couldn’t open, explain how to fix permissions
    raise SystemExit(f"Could not open webcam (index {idx}). "
                     "Enable Camera for Terminal/IDE in System Settings → Privacy & Security → Camera.")

#Main entry point for parsing args, loading the model, and running the loop
def main():
    #Set up a tiny CLI: just --model (weights) and --source (camera index or file path)
    ap = argparse.ArgumentParser()
    #Which YOLOv11 weights to load; the 'n' variant is small and quick to test
    ap.add_argument("--model", default="yolo11n.pt", help="YOLOv11 weights")
    #Camera index as a string (e.g., "0" or "1") or a file path to a video
    ap.add_argument("--source", default="0", help="Webcam index (e.g., 0/1) or path to video")
    # ctually parse the command-line arguments into an object
    args = ap.parse_args()

    #Load the YOLOv11 model from the given weights (downloads if not cached)
    model = YOLO(args.model)
    #Open the capture device or file based on --source
    cap = open_capture(args.source)

    #Create a resizable window for displaying the annotated frames
    cv2.namedWindow("YOLOv11", cv2.WINDOW_NORMAL)
    #Make the window a reasonable size for laptops/monitors
    cv2.resizeWindow("YOLOv11", 960, 540)

    #Frame-processing loop: read → run YOLO → draw → show → check for 'q'
    while True:
        #Grab the next frame from the camera/video; ok is False at EOF or on error
        ok, frame = cap.read()
        #If we can’t read a frame, stop the loop cleanly
        if not ok:
            print("No frame. Exiting.")
            break
        #Run inference on the frame; Ultralytics returns a Results list
        results = model(frame)          # inference
        # sk Ultralytics to draw the boxes/labels for us on a copy of the frame
        annotated = results[0].plot()   # draw boxes/labels
        #Show the annotated image in the window we created
        cv2.imshow("YOLOv11", annotated)
        #Poll the keyboard; if the user pressed 'q', exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Free the camera/file handle so other apps can use it
    cap.release()
    #Close the OpenCV window(s) we opened
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
