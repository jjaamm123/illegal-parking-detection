import cv2

zones = []

def draw_zone(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        zones.append((x, y))
        print(f"Point added: ({x}, {y})")

def define_zone(frame):
    cv2.namedWindow("Define Zone")
    cv2.setMouseCallback("Define Zone", draw_zone)
    cv2.imshow("Define Zone", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return zones