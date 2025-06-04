from Seq.models import Vanilla_NN
import torch


class TarNetFunctions():
    # noinspection PyAttributeOutsideInit
    # def build_networks(self):
    #     self.MLP_params_w = self.network_params['mlp_params_w']
    #     self.MLP_params_t_w = self.network_params['mlp_params_t_w']
    #     self.MLP_params_y0_w = self.network_params['mlp_params_y0_w']
    #     self.MLP_params_y1_w = self.network_params['mlp_params_y1_w']

    #     output_multiplier_t = 1 if self.binary_treatment else 2
    #     self._mlp_w = self._build_mlp(self.dim_w, self.MLP_params_w.dim_h, self.MLP_params_w, 1)
    #     self._mlp_t_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_t, self.MLP_params_t_w, output_multiplier_t)
    #     self._mlp_y0_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_y, self.MLP_params_y0_w,
    #                                      self.outcome_distribution.num_params)
    #     self._mlp_y1_w = self._build_mlp(self.MLP_params_w.dim_h, self.dim_y, self.MLP_params_y1_w,
    #                                      self.outcome_distribution.num_params)
    #     self.networks = [self._mlp_w, self._mlp_t_w, self._mlp_y0_w, self._mlp_y1_w]

    
    def __init__(self, dims, MODEL_PARAMS, treatment_distribution, outcome_distribution):
        self.dims = dims
        self.MODEL_PARAMS = MODEL_PARAMS
        self.treatment_distribution = treatment_distribution
        self.outcome_distribution = outcome_distribution
        self.build_networks()

        torch.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed_all(MODEL_PARAMS["seed"])

    def build_networks(self):
        self.networks_y = []
        self.networks = []
        self.model_x = Vanilla_NN(input_size=self.dims["dim_x"], output_size=self.dims["dim_dense_y"],
                           hidden_size=self.dims["dim_dense_y"], num_layers=self.MODEL_PARAMS["y"]["n_dense_y_x"])
        self.networks.append(self.model_x)
        self.networks_y.append(self.model_x)

        if self.MODEL_PARAMS["loss_type"] == 'separate':
            self.model_t_x = Vanilla_NN(input_size=self.dims["dim_x"], output_size=self.dims["dim_t"],
                                    hidden_size=self.dims["dim_dense_t"], num_layers=self.MODEL_PARAMS["t"]["n_dense_t"])
        else:
            self.model_t_x = Vanilla_NN(input_size=self.dims["dim_dense_y"], output_size=self.dims["dim_t"],
                                    hidden_size=self.dims["dim_dense_t"], num_layers=self.MODEL_PARAMS["t"]["n_dense_t"])
        
        if self.MODEL_PARAMS["t_already_trained"]:
            self.model_t_x.load_state_dict(torch.load(self.MODEL_PARAMS["t_model_path"]))

        self.networks.append(self.model_t_x)

        for _ in range(max(self.dims["dim_t"], 2)):
            network = Vanilla_NN(input_size=self.dims["dim_dense_y"], output_size=self.dims["dim_outcome_distribution"],
                                        hidden_size=self.dims["dim_dense_y"], num_layers=self.MODEL_PARAMS["y"]["n_dense_y_x"])
            self.networks_y.append(network)
        # if self.dims["dim_t"] < 2:
        #     self.model_y_0_x = Vanilla_NN(input_size=self.dims["dim_dense"], output_size=self.dims["dim_outcome_distribution"],
        #                             hidden_size=self.dims["dim_dense"], num_layers=self.MODEL_PARAMS["n_dense_y_x"])
        #     self.networks_y.append(self.model_y_0_x)

        #     self.model_y_1_x = Vanilla_NN(input_size=self.dims["dim_dense"], output_size=self.dims["dim_outcome_distribution"],
        #                                 hidden_size=self.dims["dim_dense"], num_layers=self.MODEL_PARAMS["n_dense_y_x"])
        #     self.networks_y.append(self.model_y_1_x)

        # print the number of trainable parameters for each network
        for network in self.networks_y:
            print(f"Number of trainable parameters for X, Y0, Y1: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
        # also for the treatment network
        print(f"Number of trainable parameters for T: {sum(p.numel() for p in self.model_t_x.parameters() if p.requires_grad)}")

    def forward_x(self, x):
        # apply relu one more time on output of model_x
        relu_output = torch.nn.functional.relu(self.model_x.forward(x))
        return relu_output
    
    def forward_t_x(self, x):
        if self.MODEL_PARAMS["loss_type"] == 'separate':
            return self.model_t_x.forward(x)
        else:
            return self.model_t_x.forward(self.forward_x(x))
        
    def forward_y_t_x(self, x, t, ret_counterfactuals=False):
        """
        :return: parameter of the conditional distribution p(y|t,w)
        """
        h_ = self.forward_x(x)
        y_total_ = [model.forward(h_) for model in self.networks_y[1:]]
        if ret_counterfactuals:
            return y_total_
        else:
            y_ = torch.stack(y_total_, dim=1)
            indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_[torch.arange(y_.size(0)), indices.long()]
            return y_

        # y0 = self.model_y_0_x.forward(h_)
        # y1 = self.model_y_1_x.forward(h_)
        # if ret_counterfactuals:
        #     return y0, y1
        # else:
        #     return y0 * (1 - t) + y1 * t
        
    def get_loss_t_x(self, x, t):
        # this is only used when loss_type is 'separate'
        """
        :return: loss of the treatment model, only used when loss_type is 'separate'
        """
        t_ = self.model_t_x.forward(x)
        loss_t = self.treatment_distribution.loss(t, t_)
        return loss_t
    
    def get_loss_y_t_x(self, x, t, y):
        # this is only used when loss_type is 'separate'
        """
        :return: loss of the outcome model, only used when loss_type is 'separate'
        """
        # h_ = self.forward_x(x)
        # y0 = self.model_y_0_x.forward(h_)
        # y1 = self.model_y_1_x.forward(h_)
        # t_reshaped = t.view(-1, 1)
        # y_ = y0 * (1 - t_reshaped) + y1 * t_reshaped
        # loss_y = self.outcome_distribution.loss(y, y_)

        h_ = self.forward_x(x)
        y_total_ = [model.forward(h_) for model in self.networks_y[1:]]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        loss_y = self.outcome_distribution.loss(y, y_)

        return loss_y
    
    def get_loss(self, x, t, y):
        """
        :return: total loss of the model, only used when loss_type is 'JOINT'
        """
        # h_ = self.forward_x(x)
        # t_ = self.model_t_x.forward(h_)
        # y0 = self.model_y_0_x.forward(h_)
        # y1 = self.model_y_1_x.forward(h_)
        # t_reshaped = t.view(-1, 1)
        # y_ = y0 * (1 - t_reshaped) + y1 * t_reshaped
        # loss_t = self.treatment_distribution.loss(t, t_)
        # loss_y = self.outcome_distribution.loss(y, y_)
        # loss = loss_t + loss_y

        h_ = self.forward_x(x)

        t_ = self.model_t_x.forward(h_)
        loss_t = self.treatment_distribution.loss(t, t_)
        
        y_total_ = [model.forward(h_) for model in self.networks_y[1:]]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        loss_y = self.outcome_distribution.loss(y, y_)

        loss = loss_t + loss_y

        return loss, loss_t, loss_y

    # def mlp_w(self, w):
    #     return self.MLP_params_w.activation(self._mlp_w(w))

    # def mlp_t_w(self, w):
    #     return self._mlp_t_w(self.mlp_w(w))

    # def mlp_y_tw(self, wt, ret_counterfactuals=False):
    #     """
    #     :param wt: concatenation of w and t
    #     :return: parameter of the conditional distribution p(y|t,w)
    #     """
    #     w, t = wt[:, :-1], wt[:, -1:]
    #     w = self.mlp_w(w)
    #     y0 = self._mlp_y0_w(w)
    #     y1 = self._mlp_y1_w(w)
    #     if ret_counterfactuals:
    #         return y0, y1
    #     else:
    #         return y0 * (1 - t) + y1 * t


    # def _get_loss(self, w, t, y):
    #     # compute w_ only once
    #     w_ = self.mlp_w(w)
    #     t_ = self._mlp_t_w(w_)
    #     if self.ignore_w:
    #         w_ = torch.zeros_like(w_)

    #     y0 = self._mlp_y0_w(w_)
    #     y1 = self._mlp_y1_w(w_)
    #     if torch.isnan(y0).any():
    #         print("y0 has nan values")
    #     if torch.isnan(y1).any():
    #         print("y1 has nan values")
    #     y_ = y0 * (1 - t) + y1 * t
    #     # check if y_ has nan values
    #     if torch.isnan(y_).any():
    #         print("y_ has nan values")
    #         print(y_, "y_")
    #     loss_t = self.treatment_distribution.loss(t, t_)
    #     loss_y = self.outcome_distribution.loss(y, y_)
    #     loss = loss_t + loss_y
    #     return loss, loss_t, loss_y