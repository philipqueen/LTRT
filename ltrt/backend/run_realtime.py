from multiprocessing import Event, Queue, Process
from pathlib import Path
import time

from aniposelib.cameras import CameraGroup

from ltrt.mock_data.mock_multiframe_payload import mock_camera_input
from ltrt.backend.realtime_pipeline import heavyweight_realtime_pipeline, lightweight_realtime_pipeline

def run_realtime(calibration_toml_path: str | Path, stop_event) -> list[Process]:
    camera_group = CameraGroup.load(str(calibration_toml_path))

    frame_payload_queue = Queue()

    processes = []

    print("creating mock camera input process")
    mock_camera_process = Process(
        target=mock_camera_input,
        args=[frame_payload_queue]
    )
    processes.append(mock_camera_process)

    output_data_queue = Queue(maxsize=1)

    print("creating lightweight realtime pipeline process")
    realtime_pipeline_process = Process(
        target=lightweight_realtime_pipeline,
        args=[camera_group, frame_payload_queue, output_data_queue, stop_event]
    )
    # print("starting heavyweight realtime pipeline process")
    # realtime_pipeline_process = Process(
    #     target=heavyweight_realtime_pipeline,
    #     args=[camera_group, frame_payload_queue, output_data_queue, stop_event]
    # )
    print("starting processes")
    mock_camera_process.start()
    realtime_pipeline_process.start()
    processes.append(realtime_pipeline_process)

    # processes sharing a multiprocessing primitive (the queues and stop event here) must start fully before function ends
    # this sleep should work, but if you get FileNotFound errors here, it may require joining the processes within this function
    time.sleep(3)

    print("finished starting realtime")
    return processes

def shutdown_realtime(processes: list[Process]) -> None:
    for process in processes:
        process.join()

if __name__ == "__main__":
    # import cProfile
    # import pstats
    stop_event = Event()
    calibration_toml_path = ".assets/freemocap_sample_data/freemocap_sample_data_camera_calibration.toml"
    processes = run_realtime(calibration_toml_path, stop_event)
    while not stop_event.is_set():
        time.sleep(0.1)
    shutdown_realtime(processes=processes)


    # with cProfile.Profile() as profile:
    #     cam_buffers, processes = run_realtime(calibration_toml_path, stop_event)
    #     shutdown_realtime(processes=processes, cam_buffers=cam_buffers, stop_event=stop_event)

    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()