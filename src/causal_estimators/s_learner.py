import torch
from Seq.models import LSTM, CNN, Vanilla_NN

class SLearnerFunctions():
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
            self.model = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"] + self.dims["dim_t"],
                                    nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_hidden_lstm"],
                                    nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_hidden_lstm"],
                                    nr_outputs=self.dims["dim_y"], masked=self.MODEL_PARAMS["masked"])
            self.networks.append(self.model)
        elif self.MODEL_PARAMS["model_type"] == "Vanilla_NN":
            self.model = Vanilla_NN(input_size=self.dims["dim_x"] + self.dims["dim_t"], output_size=self.dims["dim_y"],
                                        hidden_size=self.dims["dim_hidden_dense"], num_layers=self.MODEL_PARAMS["n_dense"])
            self.networks.append(self.model)
        
        # print the number of trainable parameters for each network
        for network in self.networks:
            print(f"Number of trainable parameters for Y: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
    
    """Functions for aggregation"""
    def forward_agg(self, x, t, ret_counterfactuals=False):
        y_ = []
        if ret_counterfactuals:
            for i in range(max(self.dims["dim_t"], 2)):
                current_t = torch.zeros(size=(t.shape[0], self.dims["dim_t"]))
                if self.dims["dim_t"] > 1:
                    current_t[range(t.shape[0]), i] = 1
                else:
                    current_t[range(t.shape[0]), :] = i
                # current_t[range(t.shape[0]), i] = 1
                x_with_t = torch.cat((x, current_t), dim=1)
                instance_y_ = self.model.forward(x_with_t)
                y_.append(instance_y_)
        else:
            x_with_t = torch.cat((x, t), dim=1)
            y_ = self.model.forward(x_with_t)
        return y_
    
    def get_loss_agg(self, x, t, y):
        x_with_t = torch.cat((x, t), dim=1)
        y_ = self.model.forward(x_with_t)
        # get the mse loss
        loss_y = torch.nn.functional.mse_loss(y, y_)
        return loss_y

    """Functions for sequential data"""
    def forward_seq(self, x_case, x_event, t, prefix_len, ret_counterfactuals=False):
        y_ = []
        if ret_counterfactuals:
            for i in range(max(self.dims["dim_t"], 2)):
                t_adjusted = torch.zeros(size=(t.shape[0], self.dims["dim_t"], x_event.shape[2]))
                if self.dims["dim_t"] > 1:
                    t_adjusted[range(t.shape[0]), i, prefix_len.long() - 1] = 1
                else:
                    t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = i
                x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
                instance_y_ = self.model.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
                y_.append(instance_y_)
        else:
            t_adjusted = torch.zeros(size=(t.shape[0], self.dims["dim_t"], x_event.shape[2]))
            t_adjusted[range(t.shape[0]), t.long(), prefix_len.long() - 1] = t
            x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
            y_ = self.model.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
        return y_

    def get_loss_seq(self, x_case, x_event, t, prefix_len, y):
        t_adjusted = torch.zeros(size=(t.shape[0], self.dims["dim_t"], x_event.shape[2]))
        t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = t
        x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
        y_ = self.model.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
        # get the mse loss, make sure that both y (batch, dim_y) and y_ (batch) are 2D tensors
        y_ = y_.squeeze(-1) if len(y_.shape) > 1 else y_
        loss_y = torch.nn.functional.mse_loss(y, y_)
        return loss_y