import multiprocessing
import numpy as np
from queue import Empty
from typing import Dict
from time import perf_counter_ns

from skellycam.core.frames.payloads.multi_frame_payload import MultiFramePayload
from skellytracker import YOLOPoseTracker, MediapipeHolisticTracker


def run_tracker(
    frame_queue: multiprocessing.Queue,
    output_data_queue: multiprocessing.Queue,
    stop_event,
):
    tracker = MediapipeHolisticTracker(
        model_complexity=0, static_image_mode=True
    )  # TODO: pass parameters into this
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        if frame is None:
            stop_event.set()
            output_data_queue.put(None)
            break
        else:
            tracker.process_image(frame)
            tracker.recorder.record(
                tracked_objects=tracker.tracked_objects
            )
            output_array = tracker.recorder.process_tracked_objects(
                image_size=frame.shape[:2]
            )
            tracker.recorder.clear_recorded_objects()
            output_data_queue.put(output_array)



def process_one_multiframe_payload(
    multiframe_payload: MultiFramePayload,
    payload_queues: Dict[int, multiprocessing.Queue],
    output_queues: Dict[int, multiprocessing.Queue],
) -> np.ndarray:
    start_send_frames = perf_counter_ns()
    for camera_id, frame_payload in multiframe_payload.frames.items():
        if frame_payload is None:
            raise RuntimeError("Frame Payload is None")
        payload_queues[camera_id].put(frame_payload.image)
    end_send_frames = perf_counter_ns()
    print(f"putting individual frames in processing queues took {(end_send_frames - start_send_frames) / 1e6} ms")
    outputs = {}
    while True:
        for camera_id, output_queue in output_queues.items():
            try:
                output = output_queue.get(timeout=0.1)
                if output is not None:
                    outputs[camera_id] = output
                else:
                    print(f"Recieved empty output from queue for camera {camera_id}")
                    raise RuntimeError("Non empty frame payload led to empty result")
            except Empty:
                pass
        if set(outputs.keys()) == set(multiframe_payload.frames.keys()):
            break

    return np.concatenate(list(outputs.values()), axis=0)[:, :, :2]
