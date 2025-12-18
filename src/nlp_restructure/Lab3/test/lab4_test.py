from pathlib import Path
import numpy as np
from datetime import datetime
from loguru import logger

# Add root directory to Python path
root_dir = Path(__file__).parents[1]

from Lab3.src.representations.word_embedder import WordEmbedder

def main():
    """Main test function."""
    
    # Setup results directory using pathlib
    results_dir = root_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / 'lab4_test_output.txt'
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write("Lab 4: WordEmbedder Test\n")
        f.write(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Initialize WordEmbedder
        try:
            embedder = WordEmbedder('glove-wiki-gigaword-100')
            f.write(f"Model loaded successfully\n")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
    
        # 1. Get vector for 'king'
        f.write("\n1. Vector for 'king':\n")
        king_vector = embedder.get_vector('king')
        if king_vector is not None:
            f.write(f"Shape: {king_vector.shape}\n")
            f.write(f"First 5 values: {king_vector[:5]}\n")
        else:
            f.write("'king' not found in vocabulary\n")
        
        # 2. Word similarities
        f.write("\n2. Word similarities:\n")
        king_queen_sim = embedder.get_similarity('king', 'queen')
        king_man_sim = embedder.get_similarity('king', 'man')
        
        if king_queen_sim and king_man_sim:
            f.write(f"king-queen similarity: {king_queen_sim:.4f}\n")
            f.write(f"king-man similarity: {king_man_sim:.4f}\n")
        
        # 3. Most similar words to 'computer'
        f.write("\n3. Most similar words to 'computer':\n")
        similar_words = embedder.get_most_similar('computer', top_n=10)
        for i, (word, sim) in enumerate(similar_words, 1):
            f.write(f"{i}. {word:<15} (similarity: {sim:.4f})\n")
        
        # 4. Document embedding
        f.write("\n4. Document embedding:\n")
        sentence = "The queen rules the country."
        doc_vector = embedder.embed_document(sentence)
        f.write(f"Input sentence: '{sentence}'\n")
        f.write(f"Document vector shape: {doc_vector.shape}\n")
        f.write(f"First 5 values: {doc_vector[:5]}\n")
        f.write(f"Is zero vector: {np.allclose(doc_vector, 0)}\n")
        
    logger.info(f"Test results saved to: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        # Save error to file
        error_file = root_dir / 'results' / 'lab4_test_error.txt'
        error_file.parent.mkdir(exist_ok=True)
        
        with error_file.open('w', encoding='utf-8') as f:
            f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n")
        logger.error(f"Error log saved to: {error_file}")
