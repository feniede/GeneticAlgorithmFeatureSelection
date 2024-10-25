
# Genetic Algorithm Feature Selection

The **GeneticAlgorithmFeatureSelection** class implements a genetic algorithm for feature selection in supervised learning tasks. This approach uses evolutionary strategies to select the optimal subset of features by maximizing a given fitness function, making it useful for high-dimensional datasets where traditional feature selection methods may struggle.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Example](#example)
- [Parallel Processing](#parallel-processing)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/GeneticAlgorithmFeatureSelection.git
cd GeneticAlgorithmFeatureSelection
```

Install any necessary dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The **GeneticAlgorithmFeatureSelection** class requires a fitness function, which evaluates each individual (or feature subset) based on a specified scoring metric. Optionally, a validation function can be included for additional evaluations of selected solutions.

### Parameters

- **size** (`int`): The population size.
- **n_feat** (`int`): The number of features in the dataset.
- **n_elite** (`int`): Number of elite individuals preserved in each generation.
- **n_parents** (`int`): Number of parents selected for crossover.
- **mutation_rate** (`float`): Mutation rate for altering feature subsets.
- **n_gen** (`int`): Total number of generations.
- **fitness_function** (`callable`): A function to evaluate individuals.
- **ff_params** (`dict`): Parameters for the fitness function.
- **n_feat_gen** (`int`, optional): Maximum features allowed per chromosome; defaults to no limit.
- **patience** (`int`, optional): Generations to wait for improvement in fitness.
- **parallel** (`bool`, optional): If `True`, runs the algorithm in parallel.
- **random_state** (`int`, optional): Seed for reproducibility.
- **validation_function** (`callable`, optional): Function to validate solutions beyond fitness.
- **vf_params** (`dict`, optional): Parameters for the validation function.
- **num_jobs** (`int`, optional): Number of parallel jobs (-1 to use all cores).

### Example

In the __main__ of the python file you can find a proper usage of the genetic algorithm. 

```python
from genetic_algorithm import GeneticAlgorithmFeatureSelection

# Example fitness function
def fitness_function(chromosome, **kwargs):
    # Example fitness evaluation using kwargs
    score = ...
    return score

# Initialize the genetic algorithm
genetic_algo = GeneticAlgorithmFeatureSelection(
    size= 200,
    n_feat=X_trains[0].shape[1],
    n_elite = 3,
    n_parents= 50, #X_trains[0].shape[1]//2,
    mutation_rate=0.05,
    n_feat_gen = 10,
    n_gen = 2000,
    fitness_function=fitness_function,
    ff_params = ff_params,
    parallel = True,
    patience = 50,
    random_state = 1234,
    validation_function = validation_function,
    vf_params = vf_params,
    num_jobs = -1)

# Run the genetic algorithm
best_chromosome, best_score, val_score = genetic_algo.generations()

print("Best feature subset:", best_chromosome)
print("Best fitness score:", best_score)
```

### Parallel Processing

To speed up the algorithm, enable parallel processing by setting `parallel=True` and adjust the `num_jobs` parameter to specify the number of CPU cores. Use `num_jobs=-1` to automatically use all available cores.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Create a new Pull Request.

## License

    Copyright 2013 Mir Ikram Uddin

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.