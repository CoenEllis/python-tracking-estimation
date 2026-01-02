"""
Kalman 2D Module

This module demonstrates how Kalman filters can predict the position of
something given 2D noisy data.
"""


import numpy as np
import matplotlib.pyplot as plt


class Kalman2D:
    def __init__(self, dt=1.0, steps=100, measurement_noise=2.0):
        """
        Initialize the Kalman filter parameters.

        Args:
            dt (float): The time interval between data points.
                Default is 1.0.
            steps (int): The amount of data received. Default is 100.
            measurement_noise (float): The max amount of noise applied to
                data. Default is 2.0.
        """
        self.dt = dt
        self.steps = steps
        self.measurement_noise = measurement_noise

        self.x = np.array([
            [0.0],  # x
            [0.0],  # y
            [0.0],  # vx
            [0.0],  # vy
        ])
        self.P = np.eye(4) * 500
        self.F = np.array([  # State transition
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])
        self.H = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0],  # measure y
        ])
        self.Q = np.eye(4) * 0.1  # Process noise
        self.R = np.eye(2) * (measurement_noise**2)  # Measurement noise

        # Generate true position and measurements
        self.true_pos = self._generate_true_position()
        self.measurements = self.true_pos + np.random.normal(
            0, self.measurement_noise, self.true_pos.shape
        )

        self.estimates = []

    def _generate_true_position(self):
        """
        Generate the true position data.

        Returns:
            np.ndarray: The true position of each data point.
        """
        true_vel = np.array([1.0, 0.5])
        pos = np.array([0.0, 0.0])
        true_pos = []

        for _ in range(self.steps):
            pos = pos + true_vel * self.dt
            true_pos.append(pos.copy())

        return np.array(true_pos)

    def predict_and_update(self):
        """Run the Kalman filter prediction and update steps."""
        for z in self.measurements:
            # Predict
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

            # Update
            z = z.reshape(2, 1)
            y = z - (self.H @ self.x)  # Innovation
            S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
            K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

            self.x = self.x + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

            self.estimates.append(self.x[:2].flatten())

    def plot_results(self):
        """Plot the true position, measurements, and Kalman estimates."""
        true_x, true_y = self.true_pos[:, 0], self.true_pos[:, 1]
        meas_x, meas_y = self.measurements[:, 0], self.measurements[:, 1]
        est = np.array(self.estimates)

        plt.plot(true_x, true_y, label="True Path")
        plt.scatter(meas_x, meas_y, s=10, alpha=0.5, label="Measurements")
        plt.plot(est[:, 0], est[:, 1], linewidth=2, label="Kalman Estimate")
        plt.legend()
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    # Script to test Kalman filter
    kalman2D = Kalman2D()
    kalman2D.predict_and_update()
    kalman2D.plot_results()
