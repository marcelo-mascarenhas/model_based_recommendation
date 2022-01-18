
from source.recommender import Recommender
import sys

def main():
    ratings = sys.argv[1]
    
    targets = sys.argv[2]

    recm_obj = Recommender(ratings, number_of_dimensions=4)
    
    recm_obj.funkSvd(epochs=40, learning_rate=0.0005, beta=0.02)

    column = recm_obj.evaluate(targets)
    
    final_df = recm_obj.save_csv(targets, column)

    print(final_df.to_csv(index=False))    
    
    
    
if __name__ == "__main__":
    main() 
    