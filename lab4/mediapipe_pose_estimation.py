"""
MediaPipe Pose Estimation Script
Processes video files and saves detected pose skeletons to CSV format.

The CSV format follows the structure: frame_number, landmark, x, y, z
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import argparse


class MediaPipePoseEstimator:
    """Class for estimating human pose using MediaPipe"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Pose estimator
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # MediaPipe Pose landmark names (33 landmarks)
        self.landmark_names = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
    
    def process_video(self, video_path, output_csv_path, visualize=False, output_video_path=None):
        """
        Process video file and extract pose landmarks
        
        Args:
            video_path: Path to input video file
            output_csv_path: Path to output CSV file
            visualize: Whether to visualize pose on video
            output_video_path: Path to save visualized video (optional)
        
        Returns:
            Number of frames processed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video file {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if visualization is enabled
        video_writer = None
        if visualize and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Open CSV file for writing
        csv_file = open(output_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
        
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        print(f"Video properties: {width}x{height} @ {fps} fps")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Convert back to BGR for OpenCV
            rgb_frame.flags.writeable = True
            frame_with_pose = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Extract and save landmarks if pose is detected
            if results.pose_landmarks:
                # Draw pose on frame if visualization is enabled
                if visualize:
                    self.mp_drawing.draw_landmarks(
                        frame_with_pose,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2
                        ),
                        self.mp_drawing.DrawingSpec(
                            color=(0, 0, 255), thickness=2
                        )
                    )
                
                # Save landmarks to CSV
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = self.landmark_names[idx]
                    csv_writer.writerow([
                        frame_count,
                        landmark_name,
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])
            
            # Write visualized frame if enabled
            if visualize and video_writer:
                video_writer.write(frame_with_pose)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        csv_file.close()
        
        if video_writer:
            video_writer.release()
        
        print(f"Processing complete. Processed {frame_count} frames.")
        print(f"Results saved to: {output_csv_path}")
        if visualize and output_video_path:
            print(f"Visualized video saved to: {output_video_path}")
        
        return frame_count


def main():
    parser = argparse.ArgumentParser(
        description='MediaPipe Pose Estimation - Extract pose landmarks from video'
    )
    parser.add_argument(
        '--input_video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize pose on video frames'
    )
    parser.add_argument(
        '--output_video',
        type=str,
        default=None,
        help='Path to save visualized video (requires --visualize)'
    )
    parser.add_argument(
        '--min_detection_confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence (default: 0.5)'
    )
    parser.add_argument(
        '--min_tracking_confidence',
        type=float,
        default=0.5,
        help='Minimum tracking confidence (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize pose estimator
    estimator = MediaPipePoseEstimator(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    # Process video
    estimator.process_video(
        args.input_video,
        args.output_csv,
        visualize=args.visualize,
        output_video_path=args.output_video
    )


if __name__ == '__main__':
    main()



