import cv2

# Wpisz IP swojego Raspberry Pi
cap = cv2.VideoCapture("tcp://192.168.126.14:8888")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # TUTAJ ODPALASZ ALGORYTM ŚLEDZENIA

    cv2.imshow("Stream z RPi 5", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break