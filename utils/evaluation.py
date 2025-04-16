import math
import matplotlib.pyplot as plt

class ModelEvaluator:
    """
    Evaluates a regression model's predictions on a dataset, computes errors, 
    and visualizes predictions vs. ground truth.
    """

    def __init__(self, predictor_fn, dataset, title=None, sample_size=250):
        """
        Initializes the evaluator.

        Args:
            predictor_fn (callable): Function that takes a datapoint and returns a predicted value.
            dataset (list): Collection of datapoints with at least `.price` and `.title` attributes.
            title (str, optional): Custom title for charts and reports.
            sample_size (int): Number of datapoints to evaluate.
        """
        self.predictor_fn = predictor_fn
        self.dataset = dataset
        self.title = title or predictor_fn.__name__.replace("_", " ").title()
        self.sample_size = min(sample_size, len(dataset))

        # Internal storage for evaluation metrics
        self.y_preds = []
        self.y_trues = []
        self.abs_errors = []
        self.squared_log_errors = []
        self.point_colors = []

    def _get_color(self, error, y_true):
        """
        Assigns a color label based on the prediction error.

        Returns:
            str: Color name ('green', 'orange', or 'red').
        """
        if error < 40 or error / y_true < 0.2:
            return "green"
        elif error < 80 or error / y_true < 0.4:
            return "orange"
        else:
            return "red"

    def _evaluate_point(self, idx):
        """
        Evaluates the prediction for a single datapoint and stores metrics.
        """
        datapoint = self.dataset[idx]
        y_true = datapoint.price
        y_pred = self.predictor_fn(datapoint)

        abs_error = abs(y_pred - y_true)
        log_error = math.log(y_true + 1) - math.log(y_pred + 1)
        sle = log_error ** 2
        color = self._get_color(abs_error, y_true)

        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)
        self.abs_errors.append(abs_error)
        self.squared_log_errors.append(sle)
        self.point_colors.append(color)

    def _plot_predictions(self, title):
        """
        Plots predicted vs. actual values with error-based color coding.
        """
        max_val = max(max(self.y_trues), max(self.y_preds))

        plt.figure(figsize=(12, 8))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6, label='Ideal Fit')
        plt.scatter(self.y_trues, self.y_preds, s=10, c=self.point_colors, alpha=0.6)
        
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(title)
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.legend()
        plt.grid(True)
        plt.show()

    def _generate_report(self):
        """
        Calculates and visualizes overall evaluation metrics.
        """
        avg_error = sum(self.abs_errors) / self.sample_size
        rmsle = math.sqrt(sum(self.squared_log_errors) / self.sample_size)
        hit_rate = sum(1 for color in self.point_colors if color == "green") / self.sample_size

        summary_title = (
            f"{self.title} | Avg Error: ${avg_error:,.2f} | "
            f"RMSLE: {rmsle:,.2f} | Hit Rate: {hit_rate * 100:.1f}%"
        )
        self._plot_predictions(summary_title)

    def run(self):
        """
        Runs the evaluation over the sample size.
        """
        for idx in range(self.sample_size):
            self._evaluate_point(idx)
        self._generate_report()

    @classmethod
    def evaluate(cls, predictor_fn, dataset, title=None, sample_size=250):
        """
        Convenience method to evaluate without explicitly instantiating.

        Args:
            predictor_fn (callable): The model prediction function.
            dataset (list): List of datapoints.
            title (str): Optional title for visualization.
            sample_size (int): How many samples to evaluate.
        """
        evaluator = cls(predictor_fn, dataset, title=title, sample_size=sample_size)
        evaluator.run()
