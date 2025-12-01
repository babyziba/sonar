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
#Mediapipe: used for additional computer vision tasks like hand tracking
import mediapipe as mp

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

    #Initiating Media Pipe
    mp_hands = mp.solutions.hands #Instantiate the Hands model
    mp_draw = mp.solutions.drawing_utils #Utility for drawing the landmarks
    mp_style = mp.solutions.drawing_styles #Predefined drawing styles
    
    #Set up the Hands classifier: static images off, max 2 hands, min detection + tracking confidence 0.5
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           model_complexity=0,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5
                           )

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
        frame = cv2.flip(frame, 1)  #Flip the frame horizontally (mirror view)

        #Hand Tracking:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convert the frame to RGB
        handResult = hands.process(rgb) #Process the frame and find hands

        #If we found any hands, loop over them and draw the landmarks
        if handResult.multi_hand_landmarks:
            h, w = frame.shape[:2] #Get height and width of the frame
            hands_list = handResult.multi_hand_landmarks 
            hd_list = handResult.multi_handedness or [] # Fallback to empty list if None
            for i, hand_landmarks in enumerate(hands_list):  
                if i < len(hd_list) and hd_list[i].classification: # Ensure classification exists
                    label = hd_list[i].classification[0].label # Get the label safely
                else:
                    label = "Unknown" # Fallback label if classification is missing

                #Drawing Hand landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys) #Bounding box coordinates
                cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0,255,0), 2) # Bounding box
                cv2.putText(frame, label, (x1, y1-15), # Text label on frame
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3) #Font Label

        #Run inference on the frame; Ultralytics returns a Results list
        results = model(frame, conf=0.65, iou=0.5) # Modified parameters for better accuracy
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
