import math
import matplotlib.pyplot as plt

# Color categories for plotting
COLOR_MAP = {"red": "red", "orange": "orange", "green": "green"}

class ModelEvaluator:
    """
    Evaluates a regression model on a dataset and visualizes performance using a color-coded scatter plot.
    """

    def __init__(self, predictor, data, title=None, size=250):
        """
        Initialize the evaluator.

        Args:
            predictor: A function that takes a data point and returns a price prediction.
            data: A list of data points with actual prices and optional titles.
            title: Optional title for the plot.
            size: Number of data points to evaluate.
        """
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size

        # Evaluation metrics
        self.y_pred = []
        self.y_true = []
        self.errors = []
        self.squared_log_errors = []
        self.colors = []

    def _assign_color(self, error, actual):
        """
        Determine color based on absolute and relative error.

        Args:
            error: Absolute prediction error.
            actual: Actual price.

        Returns:
            A string representing the color category.
        """
        if error < 40 or error / actual < 0.2:
            return "green"
        elif error < 80 or error / actual < 0.4:
            return "orange"
        else:
            return "red"

    def _evaluate_single_point(self, index):
        """
        Evaluate model performance on a single data point.

        Args:
            index: Index of the data point in the dataset.
        """
        datapoint = self.data[index]
        prediction = self.predictor(datapoint)
        actual = datapoint.price
        error = abs(prediction - actual)
        log_error = math.log(actual + 1) - math.log(prediction + 1)
        sle = log_error ** 2
        color = self._assign_color(error, actual)

        self.y_pred.append(prediction)
        self.y_true.append(actual)
        self.errors.append(error)
        self.squared_log_errors.append(sle)
        self.colors.append(color)

    def _plot_results(self, title):
        """
        Generate a scatter plot of predicted vs actual values.

        Args:
            title: Title of the plot.
        """
        plt.figure(figsize=(8, 6))
        max_val = max(max(self.y_true), max(self.y_pred))
        
        # Ideal prediction line (y = x)
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6, label='Ideal Prediction')

        # Scatter plot
        plt.scatter(self.y_true, self.y_pred, s=3, c=self.colors)

        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(title)
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.legend()
        plt.grid(True)
        plt.show()

    def _summarize_results(self):
        """
        Compute overall error metrics and display the evaluation plot.
        """
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.squared_log_errors) / self.size)
        green_hits = sum(1 for color in self.colors if color == "green")
        hit_rate = green_hits / self.size * 100

        plot_title = f"{self.title} | Avg Error = ${average_error:,.2f} | RMSLE = {rmsle:.2f} | Hit Rate = {hit_rate:.1f}%"
        self._plot_results(plot_title)

    def run(self):
        """
        Evaluate the model across all data points and visualize the results.
        """
        for i in range(self.size):
            self._evaluate_single_point(i)
        self._summarize_results()

    @classmethod
    def evaluate(cls, predictor, data):
        """
        Run a full evaluation on a given predictor and dataset.

        Args:
            predictor: The prediction function.
            data: The dataset to evaluate.
        """
        cls(predictor, data).run()
