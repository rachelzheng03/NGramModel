from NGramModel import *
from util import *
import json
import argparse

def load_train(filepath: str):
    train = []
    print("Loading train data ...")
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            method = line.strip().split()
            train.append(method)
    return train

def load_eval():
    eval = []
    print("Loading eval data ...\n")
    with open(r"./data/eval.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            method = line.strip().split()
            eval.append(method)
    return eval

def save_model(model: NGramModel, filepath:str):
    model_data = {}

    model_data["n"] = model.n
    model_data["ngram_counts"] = {json.dumps(k): v for k, v in model.ngram_counts.items()} # convert keys (tuples) to lists (so that it can be saved into a json file)
    model_data["n_minus_one_counts"] = {json.dumps(k): v for k, v in model.n_minus_1_gram_counts.items()} # convert keys to lists
    model_data["vocab"] = list(model.vocab) # convert set to list

    with open(filepath, "w") as f:
        json.dump(model_data, f, indent=4)
    print("Model data successfully saved to " + filepath +"\n")

def find_best_model(train:list, eval:list):
    model_dict = {}
    evaluation_dict = {}
    for i in range(3, 8, 2): # 3, 5, 7
        print(f"Training {i}-gram model:")
        model = NGramModel(i)
        model.train(train)
        perplexity = model.evaluate_model(eval)
        print(f"Perplexity for {i}-gram model: {perplexity} \n")
        model_dict[i] = model
        evaluation_dict[i] = perplexity

    min_perplexity_key = min(evaluation_dict, key=evaluation_dict.get)
    print(f"n={min_perplexity_key} yields the best model for the given training data\n")
    best_model = model_dict[min_perplexity_key]
    return best_model
    
def generate_predictions(test_100: list, model: NGramModel):
    test_results_dict = {}
    print("Generating model test results:")
    for i, tokenized_method in enumerate(test_100):
        try:
            n = model.n
            method_start = tokenized_method[:n]
            num_tokens_in_method = len(tokenized_method)
            generated_method = model.generate_full_method(method_start, num_tokens_in_method)
            test_results_dict[i] = generated_method
        except Exception as e:
            print(f"Error completing method {i}")
            print(tokenized_method)
            raise e
        print_progress(i+1)
    return test_results_dict

def test_model(model: NGramModel):
    print("Loading test data ...")
    test_100 = []
    test_all = []
    with open(r"./data/test_100.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            method = line.strip().split()
            test_100.append(method)

    with open(r"./data/test.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            method = line.strip().split()
            test_all.append(method)

    test_perplexity = model.evaluate_model(test_all)
    print(f"Perplexity on the test data: {test_perplexity}")

    test_results = generate_predictions(test_100, model)

    # save test results
    with open("results_provided_model.json", "w") as f:
        json.dump(test_results, f, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # one of the two is required
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", type=str, help="txt file that contains the training data")
    group.add_argument("--pretrain", type=str, help="the JSON file that contains the data of a model that has already been trained")
    
    parser.add_argument("--save", "-s", type=str, help="the JSON file you would like to save your model data to")
    args = parser.parse_args()

    if args.pretrain is None:
        train_filepath = "./data/" + args.train
        train = load_train(train_filepath)
        eval = load_eval()
        best_model = find_best_model(train, eval)

        if args.save is not None:
            saved_model_filepath = "./data/saved_models/" + args.save
            save_model(best_model, saved_model_filepath)
        test_model(best_model)
    else:
        # if pretrain then save cannot be used
        if args.save is not None:
            print("Error: -s cannot be used when --pretrain is specified.", file=sys.stderr)
            sys.exit(1)
        model = NGramModel(pretrained_data_file="./data/saved_models/" + args.pretrain)
        test_model(model)


    



