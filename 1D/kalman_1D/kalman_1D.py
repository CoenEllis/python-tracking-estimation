"""
Kalman 1D Module

This module demonstrates how Kalman filters can predict the position of
something given 1D noisy data.
"""


import numpy as np
import matplotlib.pyplot as plt


class Kalman1D:
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

        self.x = np.array([[0.0], [0.0]])  # [position, velocity]
        self.P = np.eye(2) * 500
        self.F = np.array([[1, dt], [0, 1]])  # State transition
        self.H = np.array([[1, 0]])  # Measure position only
        self.Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise
        self.R = np.array([[measurement_noise**2]])  # Measurement noise

        # Generate true position and measurements
        self.true_pos = self._generate_true_position()
        self.measurements = (self.true_pos +
                             np.random.normal(0, measurement_noise, steps))
        self.estimates = []

    def _generate_true_position(self):
        """
        Generate the true position data.

        Returns:
            np.ndarray: The true position of each data point.
        """
        true_vel = 1.0
        pos = 0.0
        true_pos = []

        for _ in range(self.steps):
            pos += true_vel * self.dt
            true_pos.append(pos)

        return np.array(true_pos)

    def predict_and_update(self):
        """Run the Kalman filter prediction and update steps."""
        for z in self.measurements:
            # Predict
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

            # Update
            y = z - (self.H @ self.x)  # Innovation
            S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
            K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

            self.x = self.x + K @ y
            self.P = (np.eye(2) - K @ self.H) @ self.P

            self.estimates.append(self.x[0, 0])

    def plot_results(self):
        """Plot the true position, measurements, and Kalman estimates."""
        plt.plot(self.true_pos, label="True Position")
        plt.scatter(range(self.steps), self.measurements,
                    label="Measurements", alpha=0.5)
        plt.plot(self.estimates, label="Kalman Estimate", linewidth=2)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Script to test Kalman filter
    kalman1D = Kalman1D()
    kalman1D.predict_and_update()
    kalman1D.plot_results()
