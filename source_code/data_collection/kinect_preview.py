import cv2
import pykinect_azure as pykinect

if __name__ == "__main__":

    # Initialize the library
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15

    # Start device and enable recording to .mkv
    video_filename = "output.mkv"
    device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

    # Create a body tracker configuration
    tracker_config = pykinect.k4abt_tracker_configuration_t()
    tracker_config.sensor_orientation = pykinect.K4ABT_SENSOR_ORIENTATION_DEFAULT
    tracker_config.tracker_processing_mode = pykinect.K4ABT_TRACKER_PROCESSING_MODE_GPU
    tracker_config.gpu_device_id = 0

    # Start body tracker
    bodyTracker = pykinect.start_body_tracker(tracker_config)

    cv2.namedWindow('Depth and Skeleton Tracking', cv2.WINDOW_NORMAL)

    while True:
        # Get capture
        capture = device.update()

        # Get body tracker frame
        body_frame = bodyTracker.update()

        # Get the color depth image from the capture
        ret_color, depth_color_image = capture.get_colored_depth_image()

        # Get the colored body segmentation
        ret_depth, body_image_color = body_frame.get_segmentation_image()

        if not ret_color or not ret_depth:
            continue

        # Combine both images (depth and body segmentation)
        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

        # Draw the skeletons on the combined image
        combined_image = body_frame.draw_bodies(combined_image)

        # Display the image
        cv2.imshow('Depth and Skeleton Tracking', combined_image)

        # Press 'q' to stop recording and close the window
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()
    device.stop()  # Stop the device
    bodyTracker.stop()  # Stop the body tracker
