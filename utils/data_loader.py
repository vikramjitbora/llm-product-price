from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from utils.items import Item


CHUNK_SIZE = 1000

# Define acceptable price range to filter outliers.
# A small number of items are extremely high-priced, which skews the distribution.
# Filtering these helps reduce bias and improve model training performance.
MIN_PRICE = 0.5
MAX_PRICE = 999.49

class ItemLoader:
    """
    A loader class to process and filter Amazon review dataset items for 
    LLM price prediction tasks.
    """

    def __init__(self, name):
        self.name = name
        self.dataset = None

    def from_datapoint(self, datapoint):
        """
        Convert a single datapoint to an Item object if it passes validation checks.

        Args:
            datapoint (dict): A single datapoint from the dataset.

        Returns:
            Item or None: The created Item object if valid, otherwise None.
        """
        try:
            price_str = datapoint['price']
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(datapoint, price)
                    return item if item.include else None
        except ValueError:
            # Skip datapoints with invalid/missing prices
            return None

    def from_chunk(self, chunk):
        """
        Process a chunk of datapoints and convert them to valid Item objects.

        Args:
            chunk (Dataset): A chunk of the dataset.

        Returns:
            list: List of valid Item objects.
        """
        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)
            if result:
                batch.append(result)
        return batch

    def chunk_generator(self):
        """
        Generator that yields chunks of the dataset to be processed.

        Yields:
            Dataset: A slice of the dataset of size CHUNK_SIZE.
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Load dataset chunks in parallel using multiple processes.

        Args:
            workers (int): Number of processes to use.

        Returns:
            list: All valid Item objects with category set.
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1

        # Process chunks in parallel using a pool of workers
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)

        # Assign category to each item after processing
        for result in results:
            result.category = self.name

        return results

    def load(self, workers=8):
        """
        Load the dataset and process it in parallel.

        Args:
            workers (int, optional): Number of parallel processes. Defaults to 8.

        Returns:
            list: Final list of processed Item objects.
        """
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)

        # Load full dataset for the given category
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_meta_{self.name}", 
            split="full", 
            trust_remote_code=True
        )

        results = self.load_in_parallel(workers)
        finish = datetime.now()

        print(
            f"Completed {self.name} with {len(results):,} datapoints "
            f"in {(finish - start).total_seconds() / 60:.1f} mins", flush=True
        )

        return results
