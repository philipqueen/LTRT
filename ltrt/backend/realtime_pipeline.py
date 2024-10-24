from multiprocessing import Queue
from typing import Dict, Optional
import numpy as np
from queue import Empty
from aniposelib.cameras import CameraGroup
from time import perf_counter_ns, sleep
import logging


from skellycam.core.frames.payloads.multi_frame_payload import MultiFramePayload

from skellytracker import YOLOPoseTracker, MediapipeHolisticTracker

from freemocap.utilities.geometry.rotate_by_90_degrees_around_x_axis import (
    rotate_by_90_degrees_around_x_axis,
)
from freemocap.core_processes.post_process_skeleton_data.post_process_skeleton import (
    post_process_data,
)
from freemocap.data_layer.recording_models.post_processing_parameter_models import (
    ProcessingParameterModel,
    PostProcessingParametersModel,
)
from freemocap.core_processes.process_motion_capture_videos.processing_pipeline_functions.anatomical_data_pipeline_functions import (
    calculate_anatomical_data,
)

from ltrt.backend.tracking_process import (
    process_one_multiframe_payload,
)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
skellytracker_logger = logging.getLogger("skellytracker")
skellytracker_logger.setLevel(logging.WARNING)


# takes roughly 250 ms per frame payload with yolo nano
# takes roughly 140 ms per frame payload with mediapipe model_complexity=0
def lightweight_realtime_pipeline(
    camera_group: CameraGroup,
    input_queue: Queue,
    tracking_payload_queues: Dict[int, Queue],
    tracking_output_queues: Dict[int, Queue],
    output_queue: Queue,
    stop_event,
):
    multiframe_payload_times = []
    queue_pull_times = []
    tracking_times = []
    triangulation_times = []
    start = perf_counter_ns()
    while not stop_event.is_set():
        # pull multiframe payload from queue
        print("-------------------------\nNew frame payload\n-------------------------")
        start_queue = perf_counter_ns()
        try:
            multiframe_payload: Optional[MultiFramePayload] = input_queue.get(
                timeout=0.1
            )
        except Empty:
            continue
        end_queue = perf_counter_ns()
        print(f"queue pull took {(end_queue - start_queue) / 1e6} ms")

        if multiframe_payload is None:
            stop_event.set()
            break
        else:
            print(
                f"received multiframe payload number {multiframe_payload.multi_frame_number} with frame count {len(multiframe_payload.frames)}"
            )
        # TODO: Frame payloads come in an arbitrary order - should we sort by Camera ID?
        # or, just access by order of camID (are we guaranteed to have 0-N? or can there be missing values?)

        print("-----------Processing multiframe payload-----------")
        start_track = perf_counter_ns()
        combined_array = process_one_multiframe_payload(
            multiframe_payload=multiframe_payload,
            payload_queues=tracking_payload_queues,
            output_queues=tracking_output_queues,
        )

        end_track = perf_counter_ns()
        print(f"Tracking took {(end_track - start_track) / 1e6} ms")

        # triangulate frame data with anipose
        start_triangulate = perf_counter_ns()
        triangulated_data = camera_group.triangulate(combined_array)
        # print(f"triangulated data shape: {triangulated_data.shape}")
        end_triangulate = perf_counter_ns()
        print(f"Triangulation took {(end_triangulate - start_triangulate) / 1e6} ms")
        # push 3d data to output queue
        # output_queue.put(triangulated_data)

        end = perf_counter_ns()
        print(
            f"Total time to process frame payload {multiframe_payload.multi_frame_number}: {(end - start) / 1e6} ms"
        )
        multiframe_payload_times.append((end - start) / 1e6)
        queue_pull_times.append((end_queue - start_queue) / 1e6)
        tracking_times.append((end_track - start_track) / 1e6)
        triangulation_times.append((end_triangulate - start_triangulate) / 1e6)

        start = perf_counter_ns()

    # should we flush the queue here?
    # i.e. if stop event is called but frames are still in the queue, should we cycle through until queue is empty?

    print("finished receiving multiframe payloads")

    # Timestamp statistics
    num_samples = 5
    print(f"Total multiframe payload:")
    print(f"\tAverage time: {np.mean(multiframe_payload_times[1:])} ms")  # throw out warmup frame
    print(f"\tMedian time: {np.median(multiframe_payload_times)} ms")
    print(f"\tFastest {num_samples} times: {np.sort(multiframe_payload_times)[:num_samples].tolist()} ms")
    print(f"\tSlowest {num_samples} times: {np.sort(multiframe_payload_times)[-num_samples:].tolist()} ms")
    print("Queue pull:")
    print(f"\tAverage time: {np.mean(queue_pull_times[1:])} ms")  # throw out warmup frame
    print(f"\tMedian time: {np.median(queue_pull_times)} ms")
    print(f"\tFastest {num_samples} times: {np.sort(queue_pull_times)[:num_samples].tolist()} ms")
    print(f"\tSlowest {num_samples} times: {np.sort(queue_pull_times)[-num_samples:].tolist()} ms")
    print("Tracking:")
    print(f"\tAverage time: {np.mean(tracking_times[1:])} ms")  # throw out warmup frame
    print(f"\tMedian time: {np.median(tracking_times)} ms")
    print(f"\tFastest {num_samples} times: {np.sort(tracking_times)[:num_samples].tolist()} ms")
    print(f"\tSlowest {num_samples} times: {np.sort(tracking_times)[-num_samples:].tolist()} ms")
    print("Triangulation:")
    print(f"\tAverage time: {np.mean(triangulation_times[1:])} ms")  # throw out warmup frame
    print(f"\tMedian time: {np.median(triangulation_times)} ms")
    print(f"\tFastest {num_samples} times: {np.sort(triangulation_times)[:num_samples].tolist()} ms")
    print(f"\tSlowest {num_samples} times: {np.sort(triangulation_times)[-num_samples:].tolist()} ms")


# Takes about 300 ms per frame group with mediapipe model_complexity=2
def heavyweight_realtime_pipeline(
    camera_group: CameraGroup,
    input_queue: Queue,
    output_queue: Queue,
    stop_event,
):
    # initialize tracker
    tracker = MediapipeHolisticTracker(model_complexity=2, static_image_mode=True)

    recording_parameter_model = ProcessingParameterModel(
        post_processing_parameters_model=PostProcessingParametersModel(
            run_butterworth_filter=False
        ),
    )
    start = perf_counter_ns()

    while not stop_event.is_set():
        # pull multiframe payload from queue
        start_queue = perf_counter_ns()
        try:
            multiframe_payload: Optional[MultiFramePayload] = input_queue.get(
                timeout=0.1
            )
        except Empty:
            continue
        end_queue = perf_counter_ns()
        print(f"queue pull took {(end_queue - start_queue) / 1e6} ms")

        if multiframe_payload is None:
            stop_event.set()
            break
        else:
            print(
                f"received multiframe payload number {multiframe_payload.multi_frame_number} with frame count {len(multiframe_payload.frames)}"
            )

        # TODO: Frame payloads come in an arbitrary order - should we sort by Camera ID?
        # or, just access by order of camID (are we guaranteed to have 0-N? or can there be missing values?)

        start_track = perf_counter_ns()
        for frame_payload in multiframe_payload.frames.values():
            if frame_payload is None:
                print(f"multiframe payload frames: {multiframe_payload.frames}")
                raise ValueError(
                    "received None frame payload"
                )  # TODO: decide what to do for incomplete payloads
            tracker.process_image(frame_payload.image)
            tracker.recorder.record(tracked_objects=tracker.tracked_objects)
        combined_array = tracker.recorder.process_tracked_objects(
            image_size=frame_payload.image.shape[:2]
        )[
            :, :, :2
        ]  # TODO: this doesn't account for images with different sizes
        tracker.recorder.clear_recorded_objects()
        print(f"combined array shape: {combined_array.shape}")

        end_track = perf_counter_ns()
        print(f"tracking took {(end_track - start_track) / 1e6} ms")

        # triangulate frame data with anipose
        start_triangulate = perf_counter_ns()
        triangulated_data = camera_group.triangulate(combined_array)
        print(f"triangulated data shape: {triangulated_data.shape}")
        end_triangulate = perf_counter_ns()
        print(f"triangulation took {(end_triangulate - start_triangulate) / 1e6} ms")
        # push 3d data to output queue
        # output_queue.put(triangulated_data)

        # add single frame index to triangulated data
        triangulated_data = np.expand_dims(triangulated_data, axis=0)

        # Need to disable logging setup in freemocap __init__ to be able to import as a library
        start_postprocessing = perf_counter_ns()
        print(f"triangulated_data_shape: {triangulated_data.shape}")
        rotated_data = rotate_by_90_degrees_around_x_axis(triangulated_data)
        # postprocessing data fails on butterworth filter, which requires at least 15 frames
        post_processed_data = post_process_data(
            recording_processing_parameter_model=recording_parameter_model,
            raw_skel3d_frame_marker_xyz=rotated_data,
            queue=None,
        )
        anatomical_data_dict = calculate_anatomical_data(
            processing_parameters=recording_parameter_model,
            skel3d_frame_marker_xyz=post_processed_data,
            queue=None,
        )

        # skipping data saving for now
        end_postprocessing = perf_counter_ns()
        print(
            f"postprocessing took {(end_postprocessing - start_postprocessing) / 1e6} ms"
        )

        end = perf_counter_ns()
        print(
            f"processed frame payload {multiframe_payload.multi_frame_number} in {(end - start) / 1e6} ms"
        )
        start = perf_counter_ns()

    # should we flush the queue here?
    # i.e. if stop event is called but frames are still in the queue, should we cycle through until queue is empty?

    print("finished receiving multiframe payloads")
