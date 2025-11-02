"""
CSV Skeleton Parser and Visualizer
Loads skeleton data from CSV files and visualizes them on images/videos.
"""

import csv
import cv2
import numpy as np
import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt


class SkeletonVisualizer:
    """Class for parsing and visualizing skeleton data from CSV files"""
    
    def __init__(self):
        """Initialize the visualizer with MediaPipe pose connections"""
        # MediaPipe Pose connections (bone structure)
        self.pose_connections = [
            # Face and head
            (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
            (9, 10),  # Mouth
            # Upper body
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (15, 17), (15, 19), (15, 21),  # Left hand
            (16, 18), (16, 20), (16, 22),  # Right hand
            # Lower body
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 24),  # Hips
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            (27, 29), (27, 31),  # Left foot
            (28, 30), (28, 32),  # Right foot
        ]
        
        # Colors for different body parts
        self.colors = {
            'head': (255, 0, 0),      # Blue
            'left_arm': (0, 255, 0),  # Green
            'right_arm': (255, 255, 0),  # Cyan
            'torso': (0, 0, 255),     # Red
            'left_leg': (255, 0, 255),  # Magenta
            'right_leg': (128, 0, 128),  # Purple
        }
    
    def load_csv(self, csv_path):
        """
        Load skeleton data from CSV file
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Dictionary mapping frame_number to list of landmarks
            Format: {frame: [(landmark_name, x, y, z), ...]}
        """
        skeletons = defaultdict(list)
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = int(row['frame_number'])
                landmark = row['landmark']
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                
                skeletons[frame_num].append((landmark, x, y, z))
        
        # Sort landmarks by frame number
        sorted_skeletons = dict(sorted(skeletons.items()))
        
        print(f"Loaded {len(sorted_skeletons)} frames from {csv_path}")
        return sorted_skeletons
    
    def get_landmark_index(self, landmark_name):
        """
        Get index of landmark from name
        
        Args:
            landmark_name: Name of the landmark
        
        Returns:
            Index of landmark (0-32)
        """
        landmark_map = {
            'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7, 'RIGHT_EAR': 8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
            'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
        }
        return landmark_map.get(landmark_name, -1)
    
    def draw_skeleton(self, image, landmarks_dict, scale_x=1.0, scale_y=1.0):
        """
        Draw skeleton on image
        
        Args:
            image: Image to draw on (numpy array)
            landmarks_dict: Dictionary mapping landmark names to (x, y, z) tuples
            scale_x: Scale factor for x coordinates
            scale_y: Scale factor for y coordinates
        
        Returns:
            Image with skeleton drawn
        """
        img_height, img_width = image.shape[:2]
        
        # Convert landmarks to indexed format
        landmarks = {}
        for landmark_name, x, y, z in landmarks_dict:
            idx = self.get_landmark_index(landmark_name)
            if idx >= 0:
                # Scale coordinates to image dimensions
                px = int(x * img_width * scale_x)
                py = int(y * img_height * scale_y)
                landmarks[idx] = (px, py)
        
        # Draw connections (bones)
        for connection in self.pose_connections:
            if connection[0] in landmarks and connection[1] in landmarks:
                pt1 = landmarks[connection[0]]
                pt2 = landmarks[connection[1]]
                
                # Determine color based on body part
                if connection[0] <= 10 or connection[1] <= 10:
                    color = self.colors['head']
                elif connection[0] in [11, 13, 15, 17, 19, 21] or connection[1] in [11, 13, 15, 17, 19, 21]:
                    color = self.colors['left_arm']
                elif connection[0] in [12, 14, 16, 18, 20, 22] or connection[1] in [12, 14, 16, 18, 20, 22]:
                    color = self.colors['right_arm']
                elif connection[0] in [11, 12, 23, 24] or connection[1] in [11, 12, 23, 24]:
                    color = self.colors['torso']
                elif connection[0] in [23, 25, 27, 29, 31] or connection[1] in [23, 25, 27, 29, 31]:
                    color = self.colors['left_leg']
                else:
                    color = self.colors['right_leg']
                
                cv2.line(image, pt1, pt2, color, 2)
        
        # Draw keypoints
        for idx, (px, py) in landmarks.items():
            cv2.circle(image, (px, py), 4, (0, 255, 255), -1)
        
        return image
    
    def visualize_frame(self, csv_path, frame_number, background_image=None, output_path=None):
        """
        Visualize skeleton for a specific frame
        
        Args:
            csv_path: Path to CSV file
            frame_number: Frame number to visualize
            background_image: Background image (optional)
            output_path: Path to save output image (optional)
        
        Returns:
            Visualization image
        """
        # Load skeleton data
        skeletons = self.load_csv(csv_path)
        
        if frame_number not in skeletons:
            print(f"Error: Frame {frame_number} not found in CSV")
            return None
        
        # Create background image if not provided
        if background_image is None:
            image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        else:
            if isinstance(background_image, str):
                image = cv2.imread(background_image)
            else:
                image = background_image.copy()
        
        # Draw skeleton
        landmarks = skeletons[frame_number]
        image = self.draw_skeleton(image, landmarks)
        
        # Add frame number text
        cv2.putText(image, f'Frame: {frame_number}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved visualization to: {output_path}")
        
        return image
    
    def visualize_video(self, csv_path, input_video_path=None, output_video_path=None, 
                        start_frame=1, end_frame=None, fps=30):
        """
        Visualize skeletons on video frames
        
        Args:
            csv_path: Path to CSV file
            input_video_path: Path to input video (optional, for background)
            output_video_path: Path to save output video
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all)
            fps: Frames per second for output video
        """
        # Load skeleton data
        skeletons = self.load_csv(csv_path)
        
        # Determine frame range
        if end_frame is None:
            end_frame = max(skeletons.keys())
        
        frames_to_process = [f for f in range(start_frame, end_frame + 1) if f in skeletons]
        
        if not frames_to_process:
            print("No frames to process")
            return
        
        # Get video properties if input video is provided
        if input_video_path and os.path.exists(input_video_path):
            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            input_fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps is None:
                fps = input_fps
            cap.release()
        else:
            width, height = 640, 480
        
        # Initialize video writer
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print(f"Visualizing {len(frames_to_process)} frames...")
        
        for frame_num in frames_to_process:
            # Get background frame if video is provided
            if input_video_path and os.path.exists(input_video_path):
                cap = cv2.VideoCapture(input_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Draw skeleton
            landmarks = skeletons[frame_num]
            frame = self.draw_skeleton(frame, landmarks)
            
            # Add frame number
            cv2.putText(frame, f'Frame: {frame_num}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if output_video_path:
                out.write(frame)
            
            if frame_num % 30 == 0:
                print(f"Processed frame {frame_num}...")
        
        if output_video_path:
            out.release()
            print(f"Saved visualization video to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description='CSV Skeleton Visualizer - Visualize pose skeletons from CSV files'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with skeleton data'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=None,
        help='Frame number to visualize (for single frame visualization)'
    )
    parser.add_argument(
        '--background_image',
        type=str,
        default=None,
        help='Background image for visualization'
    )
    parser.add_argument(
        '--input_video',
        type=str,
        default=None,
        help='Input video file (for video visualization)'
    )
    parser.add_argument(
        '--output_image',
        type=str,
        default=None,
        help='Path to save output image'
    )
    parser.add_argument(
        '--output_video',
        type=str,
        default=None,
        help='Path to save output video'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=1,
        help='Starting frame for video visualization'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=None,
        help='Ending frame for video visualization'
    )
    
    args = parser.parse_args()
    
    visualizer = SkeletonVisualizer()
    
    if args.frame is not None:
        # Single frame visualization
        visualizer.visualize_frame(
            args.csv,
            args.frame,
            background_image=args.background_image,
            output_path=args.output_image
        )
    elif args.output_video:
        # Video visualization
        visualizer.visualize_video(
            args.csv,
            input_video_path=args.input_video,
            output_video_path=args.output_video,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
    else:
        print("Please specify either --frame for single frame visualization or --output_video for video visualization")


if __name__ == '__main__':
    main()



