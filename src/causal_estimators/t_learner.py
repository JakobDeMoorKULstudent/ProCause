import torch
from Seq.models import LSTM, CNN, Vanilla_NN

class TLearnerFunctions():
    def __init__(self, dims, MODEL_PARAMS):
        self.dims = dims
        self.MODEL_PARAMS = MODEL_PARAMS
        self.build_networks()
        # set seed
        torch.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed_all(MODEL_PARAMS["seed"])

    def build_networks(self):
        self.networks = []
        if self.MODEL_PARAMS["model_type"] == 'LSTM':
            for _ in range(max(self.dims["dim_t"], 2)):
                network = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"],
                                    nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_hidden_lstm"],
                                    nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_hidden_lstm"],
                                    nr_outputs=self.dims["dim_y"], masked=self.MODEL_PARAMS["masked"])
                self.networks.append(network)
        elif self.MODEL_PARAMS["model_type"] == "Vanilla_NN":
            for _ in range(max(self.dims["dim_t"], 2)):
                network = Vanilla_NN(input_size=self.dims["dim_x"], output_size=self.dims["dim_y"],
                                            hidden_size=self.dims["dim_hidden_dense"], num_layers=self.MODEL_PARAMS["n_dense"])
                self.networks.append(network)
        
        # print the number of trainable parameters for each network
        for network in self.networks:
            print(f"Number of trainable parameters for Y: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
    
    """Functions for aggregation"""
    def forward_agg(self, x, t, ret_counterfactuals=False):
        y_total_ = [model.forward(x) for model in self.networks]
        if ret_counterfactuals:
            return y_total_
        else:
            y_ = torch.stack(y_total_, dim=1)
            indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_[torch.arange(y_.size(0)), indices.long()]
            return y_

    def get_loss_agg(self, x, t, y):
        y_total_ = [model.forward(x) for model in self.networks]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        # get the mse loss
        loss_y = torch.nn.functional.mse_loss(y, y_)
        return loss_y

    """Functions for sequential data"""
    def forward_seq(self, x_case, x_event, t, prefix_len, ret_counterfactuals=False):
        y_total_ = [model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len) for model in self.networks]
        if ret_counterfactuals:
            return y_total_
        else:
            y_ = torch.stack(y_total_, dim=1)
            indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_[torch.arange(y_.size(0)), indices.long()]
            return y_

    def get_loss_seq(self, x_case, x_event, t, prefix_len, y):
        y_total_ = [model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len) for model in self.networks]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        # get the mse loss, make sure that both y (batch, dim_y) and y_ (batch) are 2D tensors
        y_ = y_.squeeze(-1) if len(y_.shape) > 1 else y_
        loss_y = torch.nn.functional.mse_loss(y, y_)
        return loss_y