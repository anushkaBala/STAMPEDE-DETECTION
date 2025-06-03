# STAMPEDE-DETECTION
![Screenshot 2025-04-05 132327](https://github.com/user-attachments/assets/4fb2e7e9-f208-4596-a0ec-6aead1025749)
![image](https://github.com/user-attachments/assets/7c0cfea7-9d5c-4dae-ae59-d6ace3dafaf7)
![image](https://github.com/user-attachments/assets/26768cb7-940f-4030-bbdd-04616c31efcb)
“जनRAKSHAK” is an AI-powered crowd safety system that analyzes real-time movement patterns to detect dangerous overcrowding, abnormal behavior, and potential stampede risks before they escalate. By leveraging computer vision and deep learning, “जनRAKSHAK” provides a proactive solution to prevent stampedes, protect lives, and ensure safer public gatherings worldwide.
Purpose: Analyzes crowd behavior to enhance public safety.
Input: Real-time/CCTV feed or pre-recorded video.
Functionalities:
Person Detection and Tracking: Uses YOLOv8 to detect individuals in video frames, identifying them as "person" and employs `supervision. ByteTrack` to track each person across frames, assigning unique IDs for consistent monitoring.
Crowd Metrics Calculation: Computes crowd density (people per 10,000 pixels) by dividing the number of detected individuals by the frame area, and calculates average movement speed , along with speed variance to detect chaotic movement.
Behavior Prediction: Predicts crowd behavior (Calm, Aggressive, Dispersing, Stampede) using two methods: rule-based classification (e.g., high density and speed indicate a stampede) and an LSTM model that analyzes sequences of density and speed over 10 frames for temporal pattern recognition.
Visualization via Streamlit: Displays an annotated video feed with bounding boxes, tracking IDs, and behavior labels, alongside real-time metrics 
Alerts for Dangerous Behaviors: Issues warnings on the interface when aggressive behavior or a stampede is detected, enabling timely intervention to prevent incidents.
Web: Monitors large gatherings to ensure safety through proactive risk detection.
