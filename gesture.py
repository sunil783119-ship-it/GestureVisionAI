import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for handLms in result.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )

            lm = handLms.landmark

            count = 0

            # Index Finger
            if lm[8].y < lm[6].y:
                count += 1

            # Middle Finger
            if lm[12].y < lm[10].y:
                count += 1

            # Ring Finger
            if lm[16].y < lm[14].y:
                count += 1

            # Pinky Finger
            if lm[20].y < lm[18].y:
                count += 1

            # Display count
            cv2.putText(
                img,
                f"Fingers: {count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Finger Counter", img)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()