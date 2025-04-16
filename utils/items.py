from typing import Optional
from transformers import AutoTokenizer
import re

# Constants
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MIN_TOKENS = 150 
MAX_TOKENS = 160

MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7  # Maximum number of characters allowed in content

class Item:
    """
    Represents a single product datapoint used for price prediction.

    Attributes:
        title (str): Title of the product.
        price (float): Price of the product.
        category (str): Category the product belongs to.
        token_count (int): Number of tokens in the generated prompt.
        details (Optional[str]): Additional product details.
        prompt (Optional[str]): Prompt generated for model training.
        include (bool): Flag indicating whether this item meets filtering criteria.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = [
        '"Batteries Included?": "No"', '"Batteries Included?": "Yes"',
        '"Batteries Required?": "No"', '"Batteries Required?": "Yes"',
        "By Manufacturer", "Item", "Date First", "Package", ":", 
        "Number of", "Best Sellers", "Number", "Product "
    ]

    def __init__(self, data, price):
        """
        Initialize an Item instance with raw data and a given price.

        Args:
            data (dict): Product data containing fields like title, description, features, and details.
            price (float): Actual price of the product.
        """
        self.title = data['title']
        self.price = price
        self.details = None
        self.prompt = None
        self.include = False
        self.token_count = 0
        self.parse(data)

    def scrub_details(self) -> str:
        """
        Removes common noisy phrases from the details string.

        Returns:
            str: Cleaned details string.
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff: str) -> str:
        """
        Cleans up the provided text by removing unwanted characters, excessive whitespace,
        and words with both long length and digits (likely product IDs or irrelevant codes).

        Args:
            stuff (str): Raw string to clean.

        Returns:
            str: Scrubbed and filtered string.
        """
        # Remove special characters and normalize whitespace
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")

        # Filter out long alphanumeric tokens
        words = stuff.split(' ')
        select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(select)

    def parse(self, data: dict):
        """
        Processes product data into a prompt format if it meets certain length and token thresholds.

        Args:
            data (dict): Product data containing description, features, and details.
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'

        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'

        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'

        # Proceed only if the content has enough characters
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]

            # Combine title and contents, then tokenize
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text: str):
        """
        Constructs the training prompt using the cleaned content and appends the true price.

        Args:
            text (str): Cleaned and truncated product description.
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self) -> str:
        """
        Returns a version of the prompt with the price portion removed for inference/testing.

        Returns:
            str: Prompt without the actual price.
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self) -> str:
        """
        Returns a string representation of the item.

        Returns:
            str: A readable string showing product title and price.
        """
        return f"<{self.title} = ${self.price}>"
