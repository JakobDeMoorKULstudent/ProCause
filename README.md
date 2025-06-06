# ProCause
This repository provides the code for the paper *"ProCause: Generating Counterfactual Outcomes to Evaluate Prescriptive Process Monitoring Methods"*. 

The structure of the code is as follows:
```
ProCause/
|_ config/                          
|_ data/                            # Data generated (SimBank) and used (BPIC12/17) in experiments
|_ Seq/                             # Evaluator using Sequential Models
    |_ distributions/                   # Code for the assumed distributions of the Y and T variables
    |_ base.py                          # Base class for the generator/evaluator, including sample functions and statistical tests
    |_ models.py                        # Base models (e.g., LSTM) used for the causal learner architectures
    |_ seq_generator.py                 # Main generator/evaluator file for training, validation
    |_ s_learner.py                     # S-learner setup and forward functions
    |_ t_learner.py                     # T-learner setup and forward functions
    |_ tarnet.py                        # TARNet setup and forward functions
    |_ utils.py                         # utils only for the evaluator
|_ NonSeq/                          # Evaluator using Non-Sequential Models
|_ res/                             # Results of the experiments
    |_ res_main.ipynb                   # Calculate the Ensemble outputs, collect the results
    |_ res_utils.py                     # Utils for Calculating the Ensemble outputs and collecting the results
    |_ SimBank                          # Results for the SimBank simulation
    |_ bpic2012                         # Results for the BPIC12 dataset
    |_ bpic2017                         # Results for the BPIC17 dataset
    |_ all_rankings                     # All resulting rankings of the evaluated PresPM methods, originating from the true ranking, and from each learner/model setup (for each iteration)
    |_ figures                          # Figures of the results
|_ scripts/
    |_ CausalEstimators_run.py          # Train the PresPM methods (based on causal estimators) to evaluate with ProCause(/RealCause)
    |_ NonSeq_run.py                    # Tune, Train & Evaluate the PresPM methods using non-sequential models
    |_ Seq_run.py                       # Tune, Train & Evaluate the PresPM methods using sequential models
|_ SimBank/                         # The full SimBank simulator
|_ src/        
    |_ causal_estimators/               # Code for the PresPM methods to evaluate    
    |_ utils/                           # Code for the PresPM methods to evaluate       
        |_ preprocessor/                    # Code for preprocessing
        |_ simbank_eval/                    # Code for evaluating ProCause using SimBank
        |_ torch_two_sample_master/         # Package for statistical tests in the BPIC evaluation
        |_ hp_tuning.py                     # Code for hyperparameter tuning functions
        |_ tools.py                         # Code for saving/loading
        |_ utils.py                         # Code for Early Stopping
```

## Installation.
The ```requirements_all_code.txt``` file provides the necessary packages for ProCause and all experiments.
All code was written for ```python 3.11.5```.

## Experiments of the paper
Download the SimBank data for the experiments from [OneDrive](https://kuleuven-my.sharepoint.com/:f:/g/personal/jakob_demoor_kuleuven_be/Eka6L_hsm2JJvY-JE7vTHhcBytKdAJi5-D6pan229LJy-Q?e=I6Mthf). 

BPIC12 & 17 can be downloaded from https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204 and https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884 respectively. Afterwards, these datasets should be cleaned using ProCause/src/utils/preprocessor/bpic_cleaning.ipynb.

Put the data in the ```data/``` folder. 

Now, the results from the paper can be reproduced by setting the ```path``` variable in the config/config.py file to your directory and running the appropriate script. For example:

```
python scripts/Seq_run.py --config config/configs_methods/config_run_all.json --dataset SimBank --intervention_name time_contact_HQ --learners TarNet S-Learner T-Learner --delta 0.95 --policies all
```

Download the results of the experiments from [Google Drive/res](https://drive.google.com/drive/folders/1jh0M3iGL_zMG9RCPYZEXcyGICHPqbOv4?usp=sharing). 