"""Synthetic dataset for gymnastic exercises."""

import torch
import numpy as np
from torch.utils.data import Dataset


class GymnasticExerciseDataset(Dataset):
    """
    Synthetic dataset for gymnastic exercises.
    Generates skeleton sequences with different motion patterns for each exercise class.
    """

    def __init__(self, num_samples=1000, T=300, J=17, D=3, num_classes=5, split='train'):
        self.num_samples = num_samples
        self.T = T
        self.J = J
        self.D = D
        self.num_classes = num_classes
        self.split = split

        # Generate synthetic data
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        """Generate synthetic skeleton sequences with different motion patterns."""
        data = []
        labels = []

        for i in range(self.num_samples):
            label = i % self.num_classes
            labels.append(label)

            # Generate skeleton sequence based on exercise type
            skeleton = self._generate_exercise_sequence(label)
            data.append(skeleton)

        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _generate_exercise_sequence(self, exercise_class):
        """Generate a skeleton sequence for a specific exercise class."""
        # Base skeleton structure (simplified COCO format)
        # Joints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
        skeleton = np.zeros((self.T, self.J, self.D))

        # Add base pose (standing position)
        base_pose = np.array([
            [0, 0, 0],  # 0: nose
            [-0.1, 0.1, 0], [0.1, 0.1, 0],  # 1-2: eyes
            [-0.15, 0.05, 0], [0.15, 0.05, 0],  # 3-4: ears
            [-0.3, 0, 0], [0.3, 0, 0],  # 5-6: shoulders
            [-0.4, -0.3, 0], [0.4, -0.3, 0],  # 7-8: elbows
            [-0.4, -0.6, 0], [0.4, -0.6, 0],  # 9-10: wrists
            [-0.2, -0.5, 0], [0.2, -0.5, 0],  # 11-12: hips
            [-0.2, -1.0, 0], [0.2, -1.0, 0],  # 13-14: knees
            [-0.2, -1.5, 0], [0.2, -1.5, 0],  # 15-16: ankles
        ])

        # Add variation to make patterns less deterministic
        # Random phase offset, speed variation, amplitude variation
        phase_offset = np.random.uniform(0, 2 * np.pi)
        speed_mult = np.random.uniform(0.8, 1.2)  # Speed variation
        amp_mult = np.random.uniform(0.85, 1.15)  # Amplitude variation

        # Generate motion patterns based on exercise class
        for t in range(self.T):
            progress = t / self.T  # Normalized time [0, 1]

            if exercise_class == 0:  # Jumping jacks
                # Arms move up and down, legs spread and close
                phase = (2 * np.pi * progress * 2 + phase_offset) * speed_mult
                skeleton[t] = base_pose.copy()
                arm_motion = 0.3 * np.sin(phase) * amp_mult
                leg_motion = 0.2 * np.sin(phase) * amp_mult
                skeleton[t, 5:11, 1] += arm_motion  # Arms up/down
                skeleton[t, 11:17, 0] += leg_motion  # Legs spread/close

            elif exercise_class == 1:  # Squats
                # Body moves down and up
                phase = (np.pi * progress + phase_offset) * speed_mult
                skeleton[t] = base_pose.copy()
                vertical_offset = 0.3 * (1 - np.cos(phase)) * amp_mult
                skeleton[t, :, 1] -= vertical_offset  # Move down
                skeleton[t, 13:17, 1] += 0.2 * np.sin(phase) * amp_mult  # Knees bend

            elif exercise_class == 2:  # Arm circles
                # Arms rotate in circles
                phase = (2 * np.pi * progress * 3 + phase_offset) * speed_mult
                skeleton[t] = base_pose.copy()
                radius = 0.3 * amp_mult
                skeleton[t, 7, 0] = -0.4 + radius * np.cos(phase)  # Left elbow
                skeleton[t, 7, 1] = -0.3 + radius * np.sin(phase)
                skeleton[t, 8, 0] = 0.4 + radius * np.cos(phase)  # Right elbow
                skeleton[t, 8, 1] = -0.3 + radius * np.sin(phase)

            elif exercise_class == 3:  # Lunges
                # Alternating leg forward
                phase = (2 * np.pi * progress + phase_offset) * speed_mult
                skeleton[t] = base_pose.copy()
                lunge_amp = 0.4 * amp_mult
                if np.sin(phase) > 0:  # Left leg forward
                    skeleton[t, 11, 0] -= lunge_amp * np.sin(phase)  # Left hip
                    skeleton[t, 13, 0] -= lunge_amp * 1.25 * np.sin(phase)  # Left knee
                    skeleton[t, 15, 0] -= lunge_amp * 1.5 * np.sin(phase)  # Left ankle
                else:  # Right leg forward
                    skeleton[t, 12, 0] += lunge_amp * abs(np.sin(phase))  # Right hip
                    skeleton[t, 14, 0] += lunge_amp * 1.25 * abs(np.sin(phase))  # Right knee
                    skeleton[t, 16, 0] += lunge_amp * 1.5 * abs(np.sin(phase))  # Right ankle

            else:  # exercise_class == 4: Push-ups
                # Body moves up and down
                phase = (2 * np.pi * progress * 2 + phase_offset) * speed_mult
                skeleton[t] = base_pose.copy()
                # Rotate body to horizontal position
                vertical_offset = 0.2 * np.sin(phase) * amp_mult
                skeleton[t, :, 1] -= 0.5 + vertical_offset  # Lower body
                skeleton[t, 9:11, 1] += 0.3  # Hands on ground

            # Add more realistic noise (higher variance)
            skeleton[t] += np.random.normal(0, 0.05, (self.J, self.D))
            
            # Add slight temporal smoothing to make it more realistic
            if t > 0:
                skeleton[t] = 0.9 * skeleton[t] + 0.1 * skeleton[t-1]

        return skeleton

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

