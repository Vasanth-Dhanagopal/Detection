# import streamlit as st
# import time
# import os
# from ultralytics import YOLO
# import cv2
# import numpy as np

# with open('styles.css') as f:
#      st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# flag=0
# uploaded_file = st.file_uploader("Upload a Video")
# if uploaded_file is not None:
#     cwd = os.getcwd()
#     with open(os.path.join(cwd, 'uploaded_video.mp4'), 'wb') as f:
#         f.write(uploaded_file.getbuffer())
#         flag=1
# with st.sidebar:
#     st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
#     st.title("Pothole Detector")
#     choice = st.radio("Navigation", ["Upload","Predict"])

# if choice=='Predict':
#   if(flag==1):
#     video_path = 'uploaded_video.mp4'
#     video_path_out = 'output.mp4'

#     cap = cv2.VideoCapture(video_path)
#     ret, frame = cap.read()
#     H, W, _ = frame.shape
#     out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'avc1'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#     model = YOLO('pothole800.pt') 
#    #model = YOLO('cldetect.pt') 

#     threshold = 0.5
    
#     st.write("Making Detections...")
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     progress_bar = st.progress(0)

#     for i in range(total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame)[0]

#         for result in results.boxes.data.tolist():

#             x1, y1, x2, y2, score, class_id = result
#             if score > threshold:

#                 class_name = results.names[int(class_id)].upper()

#                 box_color = (0, 255, 0)  # Default color (green)
#                 # Determine color based on the class name (size)

#                 if class_name == 'SMALL':

#                     box_color = (0, 255, 0)  # Green

#                 elif class_name == 'MEDIUM':

#                     box_color = (255, 255, 0)  # Yellow

#                 elif class_name == 'LARGE':

#                     box_color = (255, 0, 0)  # Red

#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 4)

#                 cv2.putText(frame, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3, cv2.LINE_AA)

#         out.write(frame)
#         progress_bar.progress((i+1)/total_frames)

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     st.success('Detections are done!', icon="✅")
#     st.subheader('Result:-')

#     video_file = open(video_path_out, 'rb') 

#     video_bytes = video_file.read() 

#     st.video(video_bytes) 



import streamlit as st
import time
import os
from ultralytics import YOLO
import cv2
import numpy as np
import math

with open('styles.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
flag=0
uploaded_file = st.file_uploader("Upload a Video")
if uploaded_file is not None:
    cwd = os.getcwd()
    with open(os.path.join(cwd, 'uploaded_video.mp4'), 'wb') as f:
        f.write(uploaded_file.getbuffer())
        flag=1
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Pothole Detector")
    choice = st.radio("Navigation", ["Upload","Predict"])

print_flag=0
if choice=='Predict':
  if(flag==1):
    video_path = 'uploaded_video.mp4'
    video_path_out = 'output.mp4'

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'avc1'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model = YOLO('pothole1000.pt') 
   #model = YOLO('cldetect.pt') 
    count_line_position = 350
    bounding_box_dimensions = []
    offset=3
    
    st.write("Making Detections...")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        cv2.line(frame, (4, count_line_position), (1270, count_line_position), (0, 0, 255), 3)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_name = int(box.cls[0])
                box_color = (0, 255, 0)
                if class_name == 0 and (math.ceil((box.conf[0] * 100)) / 100) > 0.5:
                    box_color = (0, 255, 0)
                elif class_name == 1 and (math.ceil((box.conf[0] * 100)) / 100) > 0.5:
                    box_color = (255, 255, 0)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                if(class_name==0):
                    name_class='small'
                else:
                    name_class='medium'    
                cv2.putText(frame, name_class, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3, cv2.LINE_AA)
                # Calculate the centroid of the bounding box
                centroid_x = x1 + (x2 - x1) // 2
                centroid_y = y1 + (y2 - y1) // 2
                
                # Draw a circle at the centroid
                cv2.circle(frame, (centroid_x, centroid_y), radius=5, color=(0, 0, 255), thickness=-1)
                
                # Check if the centroid intersects the line
                if centroid_y<(count_line_position+offset) and centroid_y>(count_line_position-offset):
                    # Include class name in the bounding box dimensions
                    #bounding_box_dimensions.append((class_name, x1, y1, x2 - x1, y2 - y1))
                    bounding_box_dimensions.append((class_name,x2 - x1, y2 - y1))
                print_flag=print_flag+1
        out.write(frame)
        progress_bar.progress((i+1)/total_frames)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    st.success('Detections are done!', icon="✅")
    st.subheader('Result:-')

    video_file = open(video_path_out, 'rb') 

    video_bytes = video_file.read() 

    st.video(video_bytes) 
    
if(print_flag>0):
    for bbox in bounding_box_dimensions:
        #print(f"Class: {bbox[0]}, X: {bbox[1]}, Y: {bbox[2]}, Width: {bbox[3]}, Height: {bbox[4]}")
        print(f"Class: {bbox[0]},Width: {bbox[1]}, Height: {bbox[2]}")
    



