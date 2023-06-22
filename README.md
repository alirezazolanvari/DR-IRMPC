# DR-IRMPC
## Overview
This repository is about to share the source code of the simulation section of article [[1]](https://github.com/alirezazolanvari/DR-IRMPC/blob/main/README.md#reference). In this example, we use DR-IRMPC to steer two agents toward their targets in the presence of a randomly-moving obstacle. For more details please check [[1]](https://github.com/alirezazolanvari/DR-IRMPC/blob/main/README.md#reference).
## Prerequisites and usage
To run this code you need `Python 3.9` or higher. For executing the code follow these steps:
  1. Go to the project's directory.
  2. Install the requirements: `pip install -r requirements.txt`.
  3. Change the working directory to the source folder: `cd source`.
  4. Run the `main_multi_agent.py` file: `python main_multi_agent.py`.

The parameters of the problem can be modified in `params.py`.
## Reference
If you use this code in your research, we would appreciate your citations to the following papers:

[1] Alireza Zolanvari and Ashish Cherukuri. "Iterative risk-constrained model predictive control: A data-driven distributionally robust approach." In 2022 European Control Conference (ECC) (pp. 1578-1583). [[PDF]](https://arxiv.org/pdf/2111.12977)
