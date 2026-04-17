import cv2
import numpy as np
import math
from picamera2 import Picamera2

# --- PARAMETRY KONFIGURACYJNE ---
THRESHOLD_VALUE = 200      # Próg odcięcia tła (0-255). Przy filtrze 850nm tło będzie czarne.
MIN_AREA = 9               # Minimalna powierzchnia diody w pikselach (np. 3x3)
MAX_AREA = 400             # Maksymalna powierzchnia (odrzucanie dużych odblasków)
MERGE_RADIUS = 50          # Promień dopasowywania (w pikselach) dla "migającej" diody
RESOLUTION_W = 1280        # Szerokość obrazu
RESOLUTION_H = 800         # Wysokość obrazu

# Lista wykrytych unikalnych diod
detected_targets = []
target_id_counter = 0

def process_frame(frame):
    global detected_targets, target_id_counter
    
    # 1. Optymalizacja formatu YUV420
    # W formacie YUV420 górna część macierzy (od 0 do RESOLUTION_H) to czysta warstwa jasności (Y),
    # czyli idealny, gotowy obraz monochromatyczny. Odcinamy resztę, by nie obciążać procesora.
    gray = frame[:RESOLUTION_H, :]

    # 2. Binaryzacja - odcięcie ciemnego tła
    _, thresh = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    # 3. Szukanie konturów (plamek)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_frame_centroids = []

    # 4. Obliczanie środków ciężkości dla poprawnej wielkości plamek
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            # Momenty obrazu pozwalają wyznaczyć matematyczny środek plamki
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_frame_centroids.append((cx, cy))

    # 5. Aktualizacja globalnej listy celów (Tracking & Clustering)
    # Zwiększamy licznik "nieobecności" dla wszystkich znanych diod
    for target in detected_targets:
        target['frames_unseen'] += 1

    # Dopasowywanie nowo wykrytych plamek do istniejącej listy
    for (cx, cy) in current_frame_centroids:
        matched = False
        for target in detected_targets:
            # Odległość między punktem właśnie wykrytym, a tym zapisanym w bazie
            dist = math.hypot(target['x'] - cx, target['y'] - cy)
            
            if dist < MERGE_RADIUS:
                # Dioda została rozpoznana jako znana
                target['x'] = cx  # Aktualizujemy jej bieżącą pozycję na ekranie
                target['y'] = cy
                target['frames_unseen'] = 0 # Zerujemy licznik, bo dioda znów świeci
                matched = True
                break
        
        if not matched:
            # To całkowicie nowa dioda w kadrze
            detected_targets.append({
                'id': target_id_counter,
                'x': cx,
                'y': cy,
                'frames_unseen': 0
            })
            target_id_counter += 1

    # 6. Wizualizacja do testów z kamerą
    # Tworzymy kolorowy obraz z naszej warstwy mono tylko po to, by móc narysować kolorowe znaczniki
    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    for target in detected_targets:
        # 60 klatek przy 120 FPS = 0.5 sekundy "pamięci" dla najwolniejszej diody 2Hz
        if target['frames_unseen'] < 60:
            cv2.circle(display_img, (target['x'], target['y']), MERGE_RADIUS, (0, 255, 0), 1)
            cv2.putText(display_img, f"ID:{target['id']}", (target['x'] + 10, target['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.drawMarker(display_img, (target['x'], target['y']), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

    return display_img

# --- GŁÓWNA PĘTLA PROGRAMU DLA RASPBERRY PI 5 ---
if __name__ == "__main__":
    # Inicjalizacja kamery
    picam2 = Picamera2()
    
    # Konfiguracja rozdzielczości i formatu YUV
    config = picam2.create_video_configuration(
        {"main": {"format": "YUV420", "size": (RESOLUTION_W, RESOLUTION_H)}}
    )
    picam2.configure(config)
    picam2.start()
    
    print("System wizyjny uruchomiony. Trwa skanowanie w poszukiwaniu modułów OOK...")
    print("Wciśnij 'q' na klawiaturze (przy aktywnym oknie kamery), aby zakończyć.")
    
    try:
        while True:
            # Pobranie ramki bezpośrednio z bufora kamery (dostajemy macierz Numpy)
            frame = picam2.capture_array()
            
            # Przekazanie ramki do naszego algorytmu detekcji
            output = process_frame(frame)
            
            # Wyświetlenie wyniku na ekranie
            cv2.imshow("Detekcja OOK (RPi5 + OV9281)", output)
            
            # Wyjście klawiszem 'q' (czeka 1 ms w każdej pętli)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nPrzerwano działanie programu kombinacją klawiszy (Ctrl+C).")
        
    finally:
        # Poprawne zamknięcie strumienia wideo i okien
        picam2.stop()
        cv2.destroyAllWindows()
        
        # Raport końcowy
        print("\n======================================")
        print("Ostateczna lista zlokalizowanych diod:")
        print("======================================")
        if not detected_targets:
            print("Nie wykryto żadnych diod podczas tej sesji.")
        else:
            for t in detected_targets:
                print(f"Moduł ID: {t['id']} | Ostatnia znana pozycja -> X: {t['x']}, Y: {t['y']}")