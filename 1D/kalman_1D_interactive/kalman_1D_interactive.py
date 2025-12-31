"""
Kalman 1D Interactive Module

This module provides adjustability to variables to determine their value.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class Kalman1DInteractive:
    def __init__(self, dt=1.0, steps=50, measurement_noise=2.0):
        """
        Initialize the interactive Kalman filter.

        Args:
            dt (float): The time interval between data points.
                Default is 1.0.
            steps (int): The amount of data received. Default is 50.
            measurement_noise (float): The max amount of noise applied to
                data. Default is 2.0.
        """
        self.dt = dt
        self.steps = steps
        self.measurement_noise = measurement_noise

        # Initial Kalman state
        self.x = np.array([[0.0], [0.0]])
        self.P = np.eye(2) * 500
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])

        # Generate true position and measurements
        self.true_pos = self._generate_true_position()
        self.measurements = (self.true_pos +
                             np.random.normal(0, measurement_noise, steps))

        # Plot elements
        self.fig = None
        self.ax = None
        self.line_est = None

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

    def run_kalman(self, Q_val, R_val):
        """
        Run the Kalman filter with specified noise parameters.

        Args:
            Q_val (float): Process noise covariance value.
            R_val (float): Measurement noise covariance value.

        Returns:
            list: Estimated positions from the Kalman filter.
        """
        Q = np.array([[Q_val, 0], [0, Q_val]])
        R = np.array([[R_val]])
        x_est = self.x.copy()
        P_est = self.P.copy()
        estimates = []

        for z in self.measurements:
            # Predict
            x_est = self.F @ x_est
            P_est = self.F @ P_est @ self.F.T + Q
            # Update
            y = z - self.H @ x_est
            S = self.H @ P_est @ self.H.T + R
            K = P_est @ self.H.T @ np.linalg.inv(S)
            x_est = x_est + K @ y
            P_est = (np.eye(2) - K @ self.H) @ P_est
            estimates.append(x_est[0, 0])

        return estimates

    def _update_plot(self, val):
        """
        Callback function for slider updates.

        Args:
            val (float): The new slider value.
        """
        Q_val = self.sQ.val
        R_val = self.sR.val
        estimates = self.run_kalman(Q_val, R_val)
        self.line_est.set_ydata(estimates)
        self.fig.canvas.draw_idle()

    def create_interactive_plot(self):
        """Create and display the interactive plot with sliders."""
        # Initial plot
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        estimates = self.run_kalman(0.1, self.measurement_noise**2)
        self.ax.plot(self.true_pos, label="True Position")
        self.ax.scatter(range(self.steps), self.measurements,
                        label="Measurements", alpha=0.5)
        self.line_est, = self.ax.plot(
            estimates, label="Kalman Estimate", linewidth=2)
        self.ax.legend()

        # Sliders
        axcolor = 'lightgoldenrodyellow'
        axQ = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
        axR = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

        self.sQ = Slider(axQ, 'Q (Process Noise)', 0.0, 5.0, valinit=0.1)
        sR_init = self.measurement_noise**2
        self.sR = Slider(axR, 'R (Measurement Noise)', 0.1, 10.0,
                         valinit=sR_init)

        self.sQ.on_changed(self._update_plot)
        self.sR.on_changed(self._update_plot)

        plt.show()


if __name__ == "__main__":
    # Script to visualize and adjust values
    kalman_interactive = Kalman1DInteractive()
    kalman_interactive.create_interactive_plot()
