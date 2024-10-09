import multiprocessing
from pathlib import Path
import time
from typing import Dict, Optional
import cv2

from skellycam.core.frames.payloads.multi_frame_payload import MultiFramePayload
from skellycam.core.frames.payloads.frame_payload import FramePayload
from skellycam.core.frames.payloads.metadata.frame_metadata_enum import create_empty_frame_metadata
from skellytracker.utilities.get_video_paths import get_video_paths

class MockMultiFramePayload:
    """
    Mocks a MultiFramePayload based on sample videos for testing purposes

    Use with a context manager, like:
    with MockMultiFramePayload() as mock_payload:
        while mock_payload.current_payload is not None:
            queue.put(mock_payload.current_payload)
            time.sleep(0.3)
            mock_payload.next_frame_payload()
    """
    def __init__(self, synchronized_video_folder_path: str | Path | None = None):
        if synchronized_video_folder_path is None:
            synchronized_video_folder_path = Path(".assets/freemocap_sample_data/synchronized_videos")

        self.video_dict = self.load_video_dict(synchronized_video_folder_path)

        self.current_payload = self.create_initial_payload()
        
    def load_video_dict(self, synchronized_video_folder_path: str | Path) -> Dict[int, cv2.VideoCapture]:
        video_paths = get_video_paths(path_to_video_folder=synchronized_video_folder_path)

        return {
            int(video_path.stem.split("_")[-1][-1]): cv2.VideoCapture(str(video_path)) for video_path in video_paths
        }
    
    def close_video_dict(self):
        for video_capture in self.video_dict.values():
            video_capture.release()
    
    def create_initial_payload(self):
        initial_payload =  MultiFramePayload.create_initial(camera_ids=list(self.video_dict.keys()))
        for camera_id, video_capture in self.video_dict.items():
            ret, frame = video_capture.read()
            if not ret:
                print(f"Failed to read frame {self.current_payload.multi_frame_number} for camera {camera_id}")
                print("Closing video captures")
                self.close_video_dict()
                self.current_payload = None  # this ensures None is stuffed into Queue to signal processing is done, could be a better way to do this
                return None
            metadata = create_empty_frame_metadata(camera_id=camera_id, frame_number=0)
            frame = FramePayload.create(image=frame, metadata=metadata)

            initial_payload.add_frame(frame)
        return initial_payload

    def next_frame_payload(self) -> Optional[MultiFramePayload]:
        payload = MultiFramePayload.from_previous(previous=self.current_payload)

        for camera_id, video_capture in self.video_dict.items():
            ret, frame = video_capture.read()
            if not ret:
                print(f"Failed to read frame {self.current_payload.multi_frame_number} for camera {camera_id}")
                print("Closing video captures")
                self.close_video_dict()
                self.current_payload = None  # this ensures None is stuffed into Queue to signal processing is done, could be a better way to do this
                return None
            metadata = create_empty_frame_metadata(camera_id=camera_id, frame_number=payload.multi_frame_number)
            frame = FramePayload.create(image=frame, metadata=metadata)

            payload.add_frame(frame)

        self.current_payload = payload
        return payload
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_video_dict()

def mock_camera_input(camera_payload_queue: multiprocessing.Queue) -> None:
    with MockMultiFramePayload() as mock_payload:
        while mock_payload.current_payload is not None:
            camera_payload_queue.put(mock_payload.current_payload)
            time.sleep(0.033)
            mock_payload.next_frame_payload()

    camera_payload_queue.put(None)

    print("processed entire mock recording!")