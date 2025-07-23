import cv2
import numpy as np

def contrast_slider_window(image, contrast_results):
    def on_trackbar(val):
        adjusted = image.copy()
        for r in contrast_results:
            if r['contrast_ratio'] < val/10:
                x,y,w,h = r['box']
                cv2.rectangle(adjusted, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Contrast Fix Preview", adjusted)

    cv2.namedWindow("Contrast Fix Preview")
    cv2.createTrackbar("Min Ratio x0.1", "Contrast Fix Preview", 45, 70, on_trackbar)
    on_trackbar(45)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grid_snap_slider_window(image, off_grid_elements, grid_lines):
    def on_trackbar(val):
        adjusted = image.copy()
        thresh = val
        for x,y,w,h in off_grid_elements:
            xs = grid_lines['vertical'][:,0]
            close = xs[np.abs(xs - x) < thresh]
            for gx in close:
                cv2.line(adjusted, (int(gx), y), (int(gx), y+h), (255,0,0), 2)
        cv2.imshow("Grid Snap Preview", adjusted)

    cv2.namedWindow("Grid Snap Preview")
    cv2.createTrackbar("Grid Snap Thresh", "Grid Snap Preview", 10, 100, on_trackbar)
    on_trackbar(10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()