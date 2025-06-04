# ProCause
## How to run ProCause for SimBank
Download SimBank data from [OneDrive](https://kuleuven-my.sharepoint.com/:f:/g/personal/jakob_demoor_kuleuven_be/EuhF_qPmUGNKkR30eWVxENgBnflVca5sWTIdhrLa46d4Fw?e=ZassYK), and put it in the data/SimBank folder.

Adjust the path in config.py.

Run the scripts/ProCause_run.py file.

Current setup (using T-learner only):
- Atoms get calculated.
- Data (and atoms) are preprocessed (one_hot, scaled, missing values -100, post_padding with zero's, retaining only relevant control and treatment prefixes).
- Start training (set_up distributions, causal structure is currently T-learner, train using log_likelihood).
- Start testing (using multiple univariate and multivariate statistical tests where if LARGER than 0.05 = good)

Structure:
- main file = procause_main.py (builds on 'base' file base.py)
- causal setup = t_learner.py (builds on procause_main.py)
