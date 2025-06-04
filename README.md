# ProCause
This repository provides the code for the paper *"ProCause: Generating Counterfactual Outcomes to Evaluate Prescriptive Process Monitoring Methods"*. 

The structure of the code is as follows:
```
ProCause/
|_ config/                          
|_ data/                            # Data generated (SimBank) and used (BPIC12/17) in experiments
|_ NonSeq/                          # Evaluator using Non-Sequential Models
|_ res/                             # Results of the experiments
    |_ res_main.ipynb                   # Calculate the Ensemble, collect the results
|_ scripts/
    |_ CausalEstimators_run.py          # Train the PresPM methods (based on causal estimators) to evaluate with ProCause(/RealCause)
    |_ NonSeq_run.py                    # Tune, Train & Evaluate the PresPM methods using non-sequential models
    |_ Seq_run.py                       # Tune, Train & Evaluate the PresPM methods using sequential models
|_ NonSeq/                          # Evaluator using Sequential Models
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

Download the results of the experiments from [OneDrive](https://kuleuven-my.sharepoint.com/:f:/g/personal/jakob_demoor_kuleuven_be/Es2BSf9z7mZNuCP3Z3wS84sBdvlSpgp81gM3-OOA3iDQvg?e=p5O22r). 