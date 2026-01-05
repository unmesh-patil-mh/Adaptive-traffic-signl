import cv2

def resize_video(input_path, output_path, width=640, height=480):
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)
    
    cap.release()
    out.release()
    print("Video resized and saved successfully.")

# Example usage
resize_video('input_images/traffic_video.mp4', 'traffic_video.mp4')
