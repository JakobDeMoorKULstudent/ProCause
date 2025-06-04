from hyperopt import fmin, tpe, Trials, STATUS_OK
from Seq.seq_generator import SeqGenerator
from NonSeq.nonseq_generator import NonSeqGenerator
import wandb
from copy import deepcopy
import time
import numpy as np
from datetime import datetime

class HyperParamTuner():
    def __init__(self, MODEL_PARAMS, TUNING_PARAMS_SPACE, data, generator="ProCause", BIG_DATA=True, VSC=False, LOGGING=True, STAT_TESTS=False):
        self.MODEL_PARAMS = MODEL_PARAMS
        self.TUNING_PARAMS_SPACE = TUNING_PARAMS_SPACE
        self.generator = generator
        self.data_train = data["train"]
        self.data_val = data["val"]
        self.data_test = data["test"]
        self.prep_utils = data["prep_utils"]
        self.BIG_DATA = BIG_DATA
        self.VSC = VSC
        self.LOGGING = LOGGING
        self.MAX_EVALS = 15
        self.STAT_TESTS = STAT_TESTS

        # Initialize W&B project
        if self.LOGGING:
            wandb.login(key="1069a93ac679da3dd380d782307387f2cf40cf0f")
            wandb.init(project=generator, config=MODEL_PARAMS)
            print('Start Time in real time: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    """Tune T Model"""
    def _objective_t(self, params):
        # update model params
        # self.MODEL_PARAMS.update(params)
        self.MODEL_PARAMS["t"].update(params)
        print(self.MODEL_PARAMS["t"])
        # create generator
        if self.generator == "ProCause":
            generator = SeqGenerator(data_train=self.data_train,
                                          data_val=self.data_val,
                                          data_test=self.data_test,
                                          prep_utils=self.prep_utils,
                                          MODEL_PARAMS=self.MODEL_PARAMS)
        elif self.generator == "RealCause":
            generator = NonSeqGenerator(data_train=self.data_train,
                                            data_val=self.data_val,
                                            data_test=self.data_test,
                                            prep_utils=self.prep_utils,
                                            MODEL_PARAMS=self.MODEL_PARAMS)

        if self.LOGGING:
            wandb.config.update(params, allow_val_change=True)

        # train model
        generator.train_t()
        # evaluate model
        val_loss_t = generator.best_val_loss_t
        if val_loss_t < self.best_val_loss_t:
            self.best_val_loss_t = val_loss_t
            self.best_params_t = params
            self.best_network_t = deepcopy(generator.network_t)
            if self.STAT_TESTS:
                self.statistical_tests_best_t = generator.val(t_only=True)
            else:
                self.statistical_tests_best_t = None
            if self.LOGGING:
                wandb.log({"best_val_loss_t": val_loss_t, "statistical_tests_best_t": self.statistical_tests_best_t})

        return {'loss': val_loss_t, 'status': STATUS_OK}
    
    def tune_t(self):
        self.best_params_t = None
        self.best_val_loss_t = float('inf')
        self.statistical_tests_best_t = None
        self.best_network_t = None

        self.trials_t = Trials()
        best_params_t = fmin(fn=self._objective_t,
                            space=self.TUNING_PARAMS_SPACE,
                            algo=tpe.suggest,
                            max_evals=self.MAX_EVALS,
                            trials=self.trials_t,
                            rstate=np.random.default_rng(self.MODEL_PARAMS["seed"]))
        self.best_params_t = best_params_t
        # pass the best params and the network as well
        # Log final best parameters to W&B
        if self.LOGGING:
            wandb.run.summary["best_params_t"] = best_params_t
            wandb.run.summary["best_val_loss_t"] = self.best_val_loss_t
            wandb.run.summary["statistical_tests_best_t"] = self.statistical_tests_best_t
            print("End Time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return best_params_t, self.best_network_t
    
    """Tune Y Model"""
    def _objective_y(self, params):
        # update model params
        # self.MODEL_PARAMS.update(params)
        self.MODEL_PARAMS["y"].update(params)
        print("Current params: ", self.MODEL_PARAMS["y"])
        # create generator
        if self.generator == "ProCause":
            generator = SeqGenerator(data_train=self.data_train,
                                          data_val=self.data_val,
                                          data_test=self.data_test,
                                          prep_utils=self.prep_utils,
                                          MODEL_PARAMS=self.MODEL_PARAMS)
        elif self.generator == "RealCause":
            generator = NonSeqGenerator(data_train=self.data_train,
                                            data_val=self.data_val,
                                            data_test=self.data_test,
                                            prep_utils=self.prep_utils,
                                            MODEL_PARAMS=self.MODEL_PARAMS)

        if self.LOGGING:
            wandb.config.update(params, allow_val_change=True)

        # train model
        generator.train_y()
        # evaluate model
        val_loss_y = generator.best_val_loss_y
        if val_loss_y < self.best_val_loss_y:
            self.best_val_loss_y = val_loss_y
            self.best_params_y = params
            self.best_networks_y = deepcopy(generator.networks_y)
            if self.STAT_TESTS:
                self.statistical_tests_best_y = generator.val(y_only=True)
            else:
                self.statistical_tests_best_y = None
            # log the current time as well
            if self.LOGGING:
                wandb.log({"best_val_loss_y": val_loss_y, "statistical_tests_best_y": self.statistical_tests_best_y, "time": time.time()})
        return {'loss': val_loss_y, 'status': STATUS_OK}
    
    def tune_y(self):
        self.best_params_y = None
        self.best_val_loss_y = float('inf')
        self.statistical_tests_best_y = None
        self.best_networks_y = None

        self.trials_y = Trials()
        best_params_y = fmin(fn=self._objective_y,
                            space=self.TUNING_PARAMS_SPACE,
                            algo=tpe.suggest,
                            max_evals=self.MAX_EVALS,
                            trials=self.trials_y,
                            rstate=np.random.default_rng(self.MODEL_PARAMS["seed"]))
        self.best_params_y = best_params_y
        # pass the best params and the network as well
        if self.LOGGING:
            wandb.run.summary["best_params_y"] = best_params_y
            wandb.run.summary["best_val_loss_y"] = self.best_val_loss_y
            wandb.run.summary["statistical_tests_best_y"] = self.statistical_tests_best_y
            # log the current time as well
            wandb.log({"time": time.time()})
            print("End Time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return best_params_y, self.best_networks_y
    
    """Tune Joint Model"""
    def _objective_joint(self, params):
        # update model params
        # self.MODEL_PARAMS.update(params)
        self.MODEL_PARAMS["t"].update(params)
        self.MODEL_PARAMS["y"].update(params)
        # create generator
        if self.generator == "ProCause":
            generator = SeqGenerator(data_train=self.data_train,
                                          data_val=self.data_val,
                                          data_test=self.data_test,
                                          prep_utils=self.prep_utils,
                                          MODEL_PARAMS=self.MODEL_PARAMS)
        elif self.generator == "RealCause":
            generator = NonSeqGenerator(data_train=self.data_train,
                                            data_val=self.data_val,
                                            data_test=self.data_test,
                                            prep_utils=self.prep_utils,
                                            MODEL_PARAMS=self.MODEL_PARAMS)
        
        if self.LOGGING:
            wandb.config.update(params, allow_val_change=True)

        # train model
        generator.train_joint()
        # evaluate model
        val_loss_joint = generator.best_val_loss
        if val_loss_joint < self.best_val_loss_joint:
            self.best_val_loss_joint = val_loss_joint
            self.best_params_joint = params
            self.best_networks_joint = deepcopy(generator.networks)
            if self.STAT_TESTS:
                self.statistical_tests_best_joint = generator.val()
            else:
                self.statistical_tests_best_joint = None
            # log the current time as well
            if self.LOGGING:
                wandb.log({"best_val_loss_joint": val_loss_joint, "statistical_tests_best_joint": self.statistical_tests_best_joint, "time": time.time()})
        
        return {'loss': val_loss_joint, 'status': STATUS_OK}
    
    def tune_joint(self):
        self.best_params_joint = None
        self.best_val_loss_joint = float('inf')
        self.statistical_tests_best_joint = None
        self.best_networks_joint = None

        self.trials_joint = Trials()
        best_params_joint = fmin(fn=self._objective_joint,
                            space=self.TUNING_PARAMS_SPACE,
                            algo=tpe.suggest,
                            max_evals=self.MAX_EVALS,
                            trials=self.trials_joint,
                            rstate=np.random.default_rng(self.MODEL_PARAMS["seed"]))
        self.best_params_joint = best_params_joint
        # pass the best params and the network as well
        if self.LOGGING:
            wandb.run.summary["best_params_joint"] = best_params_joint
            wandb.run.summary["best_val_loss_joint"] = self.best_val_loss_joint
            wandb.run.summary["statistical_tests_best_joint"] = self.statistical_tests_best_joint
            # log the current time as well
            wandb.log({"time": time.time()})
            print("End Time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return best_params_joint, self.best_networks_joint