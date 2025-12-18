from datasets import load_dataset
import os
from pathlib import Path

def download_data():
    dataset_name = "zeroshot/twitter-financial-news-sentiment"
    output_dir = Path("data/twitter-financial-news-sentiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name)
        
        for split in dataset.keys():
            output_file = output_dir / f"{split}.csv"
            print(f"Saving {split} split to {output_file}...")
            dataset[split].to_csv(output_file, index=False)
            
        print("Download completed successfully!")
        print(f"Files saved in: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_data()
