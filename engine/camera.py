import cv2

cap = cv2.VideoCapture(0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output5.avi', fourcc, 30, (width, height), isColor=True)

while cap.isOpened():
    try:
        # get validity boolean and current frame
        ret, frame = cap.read()

        # if valid tag is false, loop back to start
        if not ret:
            break
        else:
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        cap.release()
        out.release()
        break