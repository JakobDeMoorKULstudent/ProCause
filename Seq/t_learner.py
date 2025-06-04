import torch
from Seq.models import LSTM, CNN, Vanilla_NN

class TLearnerFunctions():
    def __init__(self, dims, MODEL_PARAMS, treatment_distribution, outcome_distribution):
        self.dims = dims
        self.MODEL_PARAMS = MODEL_PARAMS
        self.treatment_distribution = treatment_distribution
        self.outcome_distribution = outcome_distribution
        self.build_networks()
        # set seed
        torch.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed(MODEL_PARAMS["seed"])
        torch.cuda.manual_seed_all(MODEL_PARAMS["seed"])

    def build_networks(self):
        self.networks_y = []
        if self.MODEL_PARAMS["model_type"] == 'LSTM':
            self.model_t_x = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"],
                                    nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_dense_t"],
                                    nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_lstm_t"],
                                    nr_outputs=self.dims["dim_t"], masked=self.MODEL_PARAMS["masked"])
            if self.MODEL_PARAMS["t_already_trained"]:
                self.model_t_x.load_state_dict(torch.load(self.MODEL_PARAMS["t_model_path"]))

            for _ in range(max(self.dims["dim_t"], 2)):
                network = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"],
                                    nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_dense_y"],
                                    nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_lstm_y"],
                                    nr_outputs=self.dims["dim_outcome_distribution"], masked=self.MODEL_PARAMS["masked"])
                self.networks_y.append(network)


                # self.model_y_0_x = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"],
                #                         nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_dense"],
                #                         nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_lstm"],
                #                         nr_outputs=self.dims["dim_outcome_distribution"], masked=self.MODEL_PARAMS["masked"])
                # self.networks_y.append(self.model_y_0_x)
                # self.model_y_1_x = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"],
                #                         nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_dense"],
                #                         nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_lstm"],
                #                         nr_outputs=self.dims["dim_outcome_distribution"], masked=self.MODEL_PARAMS["masked"])
                # self.networks_y.append(self.model_y_1_x)
        
        # print the number of trainable parameters for each network
        for network in self.networks_y:
            print(f"Number of trainable parameters for Y0, and Y1: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
        # also for the treatment network
        print(f"Number of trainable parameters for T: {sum(p.numel() for p in self.model_t_x.parameters() if p.requires_grad)}")

    def forward_t_x(self, x_case, x_event, prefix_len):
        return self.model_t_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
    
    def forward_y_t_x(self, x_case, x_event, t, prefix_len, ret_counterfactuals=False):
        """
        :return: parameter of the conditional distribution p(y|t,w)
        """
        y_total_ = [model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len) for model in self.networks_y]
        if ret_counterfactuals:
            return y_total_
        else:
            y_ = torch.stack(y_total_, dim=1)
            indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
            y_ = y_[torch.arange(y_.size(0)), indices.long()]
            return y_

        # y_ = []
        # if ret_counterfactuals:
        #     for model in self.networks_y:
        #         instance_y_ = model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        #         y_.append(instance_y_)
        # else:
        #     for i, row in enumerate(t):
        #         if self.dims["dim_t"] < 2:
        #             model_index = 0 if t[i] == 0 else 1
        #         else:
        #             model_index = torch.argmax(row).item()
        #         instance_y_ = self.networks_y[model_index].forward(x_case=x_case[i].unsqueeze(0), x_process=x_event[i].unsqueeze(0), prefix_len=prefix_len[i].unsqueeze(0))
        #         y_.append(instance_y_)
        #     y_ = torch.cat(y_, dim=0)

        # return y_

        # y0 = self.model_y_0_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        # y1 = self.model_y_1_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        # if ret_counterfactuals:
        #     return y0, y1
        # else:
        #     return y0 * (1 - t) + y1 * t
        
    def get_loss_t_x(self, x_case, x_event, t, prefix_len):
        t_ = self.model_t_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        loss_t = self.treatment_distribution.loss(t, t_)
        return loss_t
    
    def get_loss_y_t_x(self, x_case, x_event, t, prefix_len, y):
        y_total_ = [model.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len) for model in self.networks_y]
        y_ = torch.stack(y_total_, dim=1)
        indices = torch.argmax(t, dim=1) if self.dims["dim_t"] > 1 else t.squeeze(-1)
        y_ = y_[torch.arange(y_.size(0)), indices.long()]
        loss_y = self.outcome_distribution.loss(y, y_)

        # y_ = []
        # for i, row in enumerate(t):
        #     if self.dims["dim_t"] < 2:
        #         model_index = 0 if t[i] == 0 else 1
        #     else:
        #         model_index = torch.argmax(row).item()
        #     instance_y_ = self.networks_y[model_index].forward(x_case=x_case[i].unsqueeze(0), x_process=x_event[i].unsqueeze(0), prefix_len=prefix_len[i].unsqueeze(0))
        #     y_.append(instance_y_)
        # y_ = torch.cat(y_, dim=0)
        # loss_y = self.outcome_distribution.loss(y, y_)
        return loss_y

        # y0 = self.model_y_0_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        # y1 = self.model_y_1_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        # t_reshaped = t.view(-1, 1)
        # y_ = y0 * (1 - t_reshaped) + y1 * t_reshaped
        # loss_y = self.outcome_distribution.loss(y, y_)
        # return loss_y
    
    # def get_loss(self, x_case, x_event, t, prefix_len, y):
    #     t_ = self.model_t_x.forward(x_case, x_event, prefix_len=prefix_len)
    #     if self.MODEL_PARAMS["ignore_x"]:
    #         x_case = torch.zeros_like(x_case)
    #         x_event = torch.zeros_like(x_event)

    #     y0 = self.model_y_0_x.forward(x_case, x_event, prefix_len=prefix_len)
    #     y1 = self.model_y_1_x.forward(x_case, x_event, prefix_len=prefix_len)

    #     if torch.isnan(y0).any():
    #         print("y0 has nan values")
    #     if torch.isnan(y1).any():
    #         print("y1 has nan values")

    #     # y0/y1 has shape (nr_samples, nr_parameters_outcome_distribution), t has shape (nr_samples) --> make sure that parameters get taken if t is 0/1
    #     t_reshaped = t.view(-1, 1)
    #     y_ = y0 * (1 - t_reshaped) + y1 * t_reshaped
    #     # y_ = y0 * (1 - t) + y1 * t
    #     # check if y_ has nan values
    #     if torch.isnan(y_).any():
    #         print("y_ has nan values")
    #         print(y_, "y_")
    #     loss_t = self.treatment_distribution.loss(t, t_)
    #     loss_y = self.outcome_distribution.loss(y, y_)
    #     loss = loss_t + loss_y
    #     return loss, loss_t, loss_y