#!/usr/bin/env python3
"""
Download Wikipedia 20231101.en split from Hugging Face Hub
and format it as delimited text file for stfo-colbert.

Be aware that this dataset is large (~20 GB) and may take a while to download.
"""

from datasets import load_dataset


def main():
    print("Loading Wikipedia dataset (20231101.en split)...")
    print("This may take a while as the dataset is large...")

    # Load the Wikipedia dataset
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,  # Use streaming to avoid loading everything into memory
    )

    # Shuffle the dataset
    print("Shuffling dataset...")
    dataset = dataset.shuffle(
        seed=42, buffer_size=100000
    )  # Good for building index centroids of large datasets

    output_file = "wikipedia_20231101_en_shuffled.txt"
    delimiter = "\n\n--------\n\n"

    print(f"Writing to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            # Get the text column
            text = example["text"]

            # Remove any delimiter occurrences inside the document
            text = text.replace(delimiter, " ")

            # Write the document
            f.write(text)

            # Add delimiter after each document (except we'll handle the last one naturally)
            f.write(delimiter)

            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} documents...")

    print(f"Done! Wrote to {output_file}")


if __name__ == "__main__":
    main()
