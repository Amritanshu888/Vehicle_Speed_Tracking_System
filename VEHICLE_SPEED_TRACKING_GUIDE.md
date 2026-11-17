# Vehicle Speed Tracking System: Code and Workflow Guide

This guide explains how the project detects vehicles in a video, keeps track of each vehicle across frames, estimates its speed, and displays the result.

## 1. Overview

The application combines four main pieces:

1. **Tkinter** provides the desktop interface for selecting a video and starting or stopping tracking.
2. **YOLOv8** detects objects in each video frame and supplies bounding boxes, class labels, and confidence scores.
3. **SORT** associates detections from frame to frame and assigns a persistent ID to each tracked object.
4. **A pixel-to-meter calculation** estimates speed from how far a tracked bounding-box center moves between consecutive frames.

The main workflow is implemented in `vehicle_speed_tracking.py`. `sort.py` contains the SORT tracker. `classes.txt` maps model class indexes to names. `saving_yolo.py` contains separate example code for training or obtaining YOLO weights; it is not imported by the tracking application.

## 2. Starting the application

When `vehicle_speed_tracking.py` runs, it:

- Imports Tkinter, OpenCV, NumPy, Pillow, YOLO, and the `Sort` class.
- Defines the calibration constants and initializes global state.
- Loads `yolov8n.pt` as the YOLO model.
- Reads the class-name list from `classes.txt`.
- Creates the Tkinter window, video canvas, and buttons.
- Starts Tkinter's event loop with `root.mainloop()`.

The interface offers:

- **Load Video**: opens a file picker for MP4 or AVI videos.
- **Start Tracking**: starts `start_tracking` on a background thread.
- **Stop Tracking**: sets a shared flag so the tracking loop can finish.
- **View Logs**: opens a separate text window for tracking messages.
- **Exit**: stops the application.

The application expects its model file and `classes.txt` to be available relative to the process's current working directory, because they are loaded using relative paths.

## 3. Vehicle detection

The `start_tracking` function is the main processing loop:

1. It checks that a video was selected, resets the stop flag, and creates a new SORT tracker.
2. It opens the selected video with `cv2.VideoCapture`.
3. It reads frames until the video ends or tracking is stopped.
4. Each frame is resized to **800 x 450 pixels**.
5. The YOLO model runs on the resized frame.
6. For each detected box, the code reads:
   - `box.xyxy`: the top-left and bottom-right pixel coordinates `(x1, y1, x2, y2)`.
   - `box.conf`: the detector's confidence score.
   - `box.cls`: the detected class index, looked up in `classes.txt`.
7. Only detections whose class is `car`, `truck`, or `bus` and whose confidence is greater than `0.6` are kept.
8. The kept detections are assembled into a NumPy array with rows in this format:

   ```text
   [x1, y1, x2, y2, confidence]
   ```

The confidence threshold is a filter: detections at or below `0.6` are not passed to SORT. The class list includes many COCO classes, but this application only uses the three vehicle categories above.

## 4. How SORT keeps vehicle IDs stable

YOLO detects objects independently in each frame. A detection by itself does not say whether it is the same vehicle seen in the previous frame. The call `tracker.update(detections)` delegates that association to SORT.

In `sort.py`, SORT works as follows:

1. **Predict existing boxes.** Each `KalmanBoxTracker` uses a Kalman filter to predict where its bounding box should appear in the next frame. The filter models the box center, area, aspect ratio, and their changes over time.
2. **Compare predictions with detections.** `iou_batch` calculates intersection over union (IoU), a measure of overlap between two boxes.
3. **Assign detections to tracks.** `associate_detections_to_trackers` matches boxes using their IoU. It uses a direct match when the overlaps are unambiguous; otherwise it uses a linear assignment algorithm (the optional `lap` package, or SciPy's `linear_sum_assignment` fallback).
4. **Update or create tracks.** Matched trackers receive the new detection. Any unmatched detection starts a new tracker and receives an ID.
5. **Keep or remove tracks.** The application constructs SORT with `max_age=30`, so a tracker can remain alive for up to 30 frames without a matched detection. SORT's default `min_hits=3` generally requires repeated detections before reporting a new track, with a special allowance during the first few frames.

`Sort.update` returns rows in this form:

```text
[x1, y1, x2, y2, object_id]
```

The ID is an identity for a track, not a vehicle class. SORT adds one to its internal counter so IDs start at a positive value. A track can still be lost or confused if detections disappear, overlap heavily, or change position abruptly.

## 5. How the speed estimate is calculated

The project defines its calibration and timing constants near the top of `vehicle_speed_tracking.py`:

```text
CALIBRATION_DISTANCE = 5 meters
CALIBRATION_PIXELS   = 200 pixels
PIXEL_TO_METER       = 5 / 200 = 0.025 meters per pixel
FRAME_RATE           = 30 frames per second
```

For every returned track, the code calculates the center of its bounding box:

```text
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2
```

It saves the previous center for that track ID in `prev_positions`. If that ID was seen in the preceding processed frame, the code computes the Euclidean distance between the two centers:

```text
distance_pixels = sqrt((center_x - previous_x)^2
                     + (center_y - previous_y)^2)
```

It then converts that movement into an estimated speed:

```text
frame_time = 1 / FRAME_RATE
speed_kmph = (distance_pixels * PIXEL_TO_METER) / frame_time * 3.6
```

The factor `3.6` converts meters per second to kilometers per hour. For example, if the center moves 4 pixels in one processed frame, the code estimates `4 * 0.025 = 0.1` meters of movement. At the hard-coded 30 frames per second, that is `0.1 / (1/30) = 3` m/s, or **10.8 km/h**.

When a track ID is first seen, it has no previous center, so the displayed speed is set to `0`. The current center and `time.time()` value are then stored for the next iteration.

## 6. Display and logging

For each tracked box, the main loop:

- Draws a green rectangle with `cv2.rectangle`.
- Writes the track ID and estimated speed on the frame with `cv2.putText`.
- Converts the frame from OpenCV's BGR color order to RGB.
- Schedules an update of the Tkinter canvas using `canvas.after`.
- Logs an ID, class, and speed message in the log window when it is open.

`update_canvas` puts the current image on the canvas and keeps a reference to it as `canvas.image`; keeping the reference prevents Tkinter's image object from being garbage-collected while it is displayed. The `info_label` argument is passed into `start_tracking` but is not currently used.

The loop releases the video capture and logs a stop message when it finishes. The stop button sets `stop_tracking`; the loop checks that flag between frames. Pressing `q` is also checked with `cv2.waitKey`, although the video is displayed in the Tkinter canvas rather than an OpenCV window.

## 7. Accuracy and interpretation

The displayed value is an **image-based estimate**, not a calibrated speed measurement guaranteed to match a speedometer or radar device. Important constraints in the current code are:

- **The calibration is assumed, not measured per video.** The code assumes that 200 pixels always represent 5 meters. That relationship must be valid for the video and the part of the image where the vehicle is moving.
- **Perspective changes pixel scale.** In a typical road video, the same real-world distance occupies fewer pixels farther from the camera. One fixed pixel-to-meter ratio therefore cannot accurately measure vehicles at different depths. Camera angle, lens distortion, and vehicle motion toward or away from the camera also affect the estimate.
- **The frame rate is fixed at 30.** `frame_time` is set from `FRAME_RATE`, not from the video's actual frame rate or timestamps. If the source is not 30 FPS, or frames are skipped, the estimated speed will be biased.
- **`prev_time` is stored but not used.** The speed calculation does not divide by the measured elapsed time between the two observations; it always uses `1/30` second.
- **The bounding-box center is only a proxy for vehicle position.** Box movement can vary because of detection jitter, occlusion, changing box size, or tracking mistakes. No smoothing of the resulting speed is applied.
- **The logged class is not stored per track.** `object_detected` is assigned while iterating over YOLO detections, but the tracker output does not include a class. As a result, the log message's class can be the last class processed in that frame rather than the class belonging to the displayed track. The box overlay itself shows the ID and speed, not the class.

For trustworthy real-world measurements, the camera view needs a suitable calibration (often mapping image coordinates to a road plane), and speed should use reliable timestamps or the actual frame rate. The estimate should be validated against known distances and speeds before being used for enforcement or other high-stakes decisions.

## 8. Supporting files

- **`vehicle_speed_tracking.py`**: GUI, video input, detection filtering, speed estimate, drawing, and logs.
- **`sort.py`**: IoU matching, Kalman-filter box prediction, track IDs, and track lifecycle.
- **`classes.txt`**: class-index-to-name list used to interpret YOLO class IDs.
- **`yolov8n.pt`**: model weights loaded by the tracking script.
- **`saving_yolo.py`**: standalone examples for training a YOLO model and saving weights, choosing a project output directory, or loading pretrained weights. The live tracker currently loads `yolov8n.pt` directly and does not invoke this script.
- **`requirements.txt`**: present dependency list. The tracking script also imports packages such as OpenCV, NumPy, Ultralytics, and Pillow; verify that its runtime environment has all imports required by the application.
