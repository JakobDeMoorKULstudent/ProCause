import torch
from Seq.models import LSTM, CNN, Vanilla_NN

class TarNetFunctions():
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
            self.model_x = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"],
                                nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_hidden_lstm"],
                                nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_hidden_lstm"],
                                nr_outputs=self.dims["dim_hidden_dense"], masked=self.MODEL_PARAMS["masked"])
            self.networks.append(self.model_x)

            for _ in range(max(self.dims["dim_t"], 2)):
                network = Vanilla_NN(input_size=self.dims["dim_hidden_dense"], output_size=self.dims["dim_y"],
                                        hidden_size=self.dims["dim_hidden_dense"], num_layers=self.MODEL_PARAMS["n_dense"])
                self.networks.append(network)

        elif self.MODEL_PARAMS["model_type"] == "Vanilla_NN":
            self.model_x = Vanilla_NN(input_size=self.dims["dim_x"], output_size=self.dims["dim_hidden_dense"],
                                        hidden_size=self.dims["dim_hidden_dense"], num_layers=self.MODEL_PARAMS["n_dense"])
            self.networks.append(self.model_x)

            for _ in range(max(self.dims["dim_t"], 2)):
                network = Vanilla_NN(input_size=self.dims["dim_hidden_dense"], output_size=self.dims["dim_y"],
                                        hidden_size=self.dims["dim_hidden_dense"], num_layers=self.MODEL_PARAMS["n_dense"])
                self.networks.append(network)

        # print the number of trainable parameters for each network
        for network in self.networks:
            print(f"Number of trainable parameters for Y: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
    
    """Functions for aggregation"""
    def forward_x_agg(self, x):
        # apply relu one more time on output of model_x
        relu_output = torch.nn.functional.relu(self.model_x.forward(x))
        return relu_output

    def forward_agg(self, x, t, ret_counterfactuals=False):
        h_ = self.forward_x_agg(x)
        y_total_ = [model.forward(h_) for model in self.networks[1:]]
        if ret_counterfactuals:
            return y_total_
        else:
            y_ = torch.stack(y_total_, dim=1)
            indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_total_[indices]
            return y_

    def get_loss_agg(self, x, t, y):
        h_ = self.forward_x_agg(x)
        y_total_ = [model.forward(h_) for model in self.networks[1:]]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        # get the mse loss
        loss_y = torch.nn.functional.mse_loss(y, y_)
        return loss_y

    """Functions for sequential data"""
    def forward_seq(self, x_case, x_event, t, prefix_len, ret_counterfactuals=False):
        h_ = self.model_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        y_total_ = [model.forward(h_) for model in self.networks[1:]]
        if ret_counterfactuals:
            return y_total_
        else:
            y_ = torch.stack(y_total_, dim=1)
            indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_[torch.arange(y_.size(0)), indices.long()]
            return y_

    def get_loss_seq(self, x_case, x_event, t, prefix_len, y):
        h_ = self.model_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        y_total_ = [model.forward(h_) for model in self.networks[1:]]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        # get the mse loss, make sure that both y (batch, dim_y) and y_ (batch) are 2D tensors
        y_ = y_.squeeze(-1) if len(y_.shape) > 1 else y_
        loss_y = torch.nn.functional.mse_loss(y, y_)
        return loss_y