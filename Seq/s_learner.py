import torch
from Seq.models import LSTM, CNN, Vanilla_NN

class SLearnerFunctions():
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

            self.model_y_t_x = LSTM(input_size_case=self.dims["dim_x_case"], input_size_process=self.dims["dim_x_event"] + self.dims["dim_t"],
                                    nr_dense_layers=self.MODEL_PARAMS["n_dense_in_lstm"], dense_width=self.dims["dim_dense_y"],
                                    nr_lstm_layers=self.MODEL_PARAMS["n_lstm"], lstm_size=self.dims["dim_lstm_y"],
                                    nr_outputs=self.dims["dim_outcome_distribution"], masked=self.MODEL_PARAMS["masked"])
            self.networks_y.append(self.model_y_t_x)
        
        # print the number of trainable parameters for each network
        for network in self.networks_y:
            print(f"Number of trainable parameters for Y: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")
        # also for the treatment network
        print(f"Number of trainable parameters for T: {sum(p.numel() for p in self.model_t_x.parameters() if p.requires_grad)}")

    def forward_t_x(self, x_case, x_event, prefix_len):
        return self.model_t_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
    
    def forward_y_t_x(self, x_case, x_event, t, prefix_len, ret_counterfactuals=False):
        """
        :return: parameter of the conditional distribution p(y|t,w)
        """
        y_ = []
        if ret_counterfactuals:
            for i in range(max(self.dims["dim_t"], 2)):
                # TODO: CHECK WHETHER CORRECT FOR TIME_CONTACT_HQ
                t_adjusted = torch.zeros(size=(t.shape[0], self.dims["dim_t"], x_event.shape[2]))
                if self.dims["dim_t"] > 1:
                    t_adjusted[range(t.shape[0]), i, prefix_len.long() - 1] = 1
                else:
                    t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = i
                # t_adjusted[range(t.shape[0]), i, prefix_len.long() - 1] = 1
                x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
                instance_y_ = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
                y_.append(instance_y_)

                # t_i = torch.zeros(size=(self.dims["dim_t"], x_event.shape[2]))
                # t_i[i, prefix_len.long() - 1] = 1
                # x_event_i_with_t_i = torch.cat((x_event, t_i), dim=0)
                # instance_y_ = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_i_with_t_i, prefix_len=prefix_len)
                # y_.append(instance_y_)
        else:
            t_adjusted = torch.zeros(size=(t.shape[0], self.dims["dim_t"], x_event.shape[2]))
            t_adjusted[range(t.shape[0]), t.long(), prefix_len.long() - 1] = t
            x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
            y_ = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)

            # for i, row in enumerate(t):
            #     # grab the correct x_case, x_event, prefix_len, and t for the current row
            #     t_i = torch.zeros(size=(self.dims["dim_t"], x_event.shape[2]))
            #     t_i[i, prefix_len.long() - 1] = row
            #     x_event_i_with_t_i = torch.cat((x_event, t_i), dim=0)
            #     # x_event_i_with_t_i = torch.cat((x_event[i], t[i]), dim=1)
            #     instance_y_ = self.model_y_t_x.forward(x_case=x_case[i].unsqueeze(0), x_process=x_event_i_with_t_i.unsqueeze(0), prefix_len=prefix_len[i].unsqueeze(0))
            #     y_.append(instance_y_)
            # y_ = torch.cat(y_, dim=0)
        
        return y_

        # # Make sure t1 has a 1 at the prefix_len position, zeros elsewhere (before prefix_len and after prefix_len), and make it the same sequence length as x_event
        # t1 = torch.zeros(size=(x_event.shape[0], self.dims["dim_t"], x_event.shape[2]))
        # t1[:, :, prefix_len.long() - 1] = 1

        # # Make sure t0 has a 0 at the prefix_len position, zeros elsewhere (before prefix_len and after prefix_len)
        # t0 = torch.zeros(size=(x_event.shape[0], self.dims["dim_t"], x_event.shape[2]))

        # # Concatenate t to x_event
        # x_event_1 = torch.cat((x_event, t1), dim=1)
        # x_event_0 = torch.cat((x_event, t0), dim=1)

        # y0 = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_0, prefix_len=prefix_len)
        # y1 = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_1, prefix_len=prefix_len)

        # if ret_counterfactuals:
        #     return y0, y1
        # else:
        #     return y0 * (1 - t) + y1 * t
        
    def get_loss_t_x(self, x_case, x_event, t, prefix_len):
        t_ = self.model_t_x.forward(x_case=x_case, x_process=x_event, prefix_len=prefix_len)
        loss_t = self.treatment_distribution.loss(t, t_)
        return loss_t

    def get_loss_y_t_x(self, x_case, x_event, t, prefix_len, y):
        t_adjusted = torch.zeros(size=(t.shape[0], self.dims["dim_t"], x_event.shape[2]))
        t_adjusted[range(t.shape[0]), :, prefix_len.long() - 1] = t
        x_event_with_t = torch.cat((x_event, t_adjusted), dim=1)
        y_ = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_with_t, prefix_len=prefix_len)
        loss_y = self.outcome_distribution.loss(y, y_)

        # y_ = []
        # for i, row in enumerate(t):
        #     # grab the correct x_case, x_event, prefix_len, and t for the current row
        #     t_i = torch.zeros(size=(self.dims["dim_t"], x_event.shape[2]))
        #     t_i[:, prefix_len[i].long() - 1] = row
        #     x_event_i_with_t_i = torch.cat((x_event[i], t_i), dim=0)
        #     # x_event_i_with_t_i = torch.cat((x_event[i], t[i]), dim=1)
        #     instance_y_ = self.model_y_t_x.forward(x_case=x_case[i].unsqueeze(0), x_process=x_event_i_with_t_i.unsqueeze(0), prefix_len=prefix_len[i].unsqueeze(0))
        #     y_.append(instance_y_)
        # y_ = torch.cat(y_, dim=0)
        # loss_y = self.outcome_distribution.loss(y, y_)
        return loss_y

        # t1 = torch.zeros(size=(x_event.shape[0], self.dims["dim_t"], x_event.shape[2]))
        # t1[:, :, prefix_len.long() - 1] = 1
        
        # t0 = torch.zeros(size=(x_event.shape[0], self.dims["dim_t"], x_event.shape[2]))
        
        # x_event_1 = torch.cat((x_event, t1), dim=1)
        # x_event_0 = torch.cat((x_event, t0), dim=1)
        
        # y0 = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_0, prefix_len=prefix_len)
        # y1 = self.model_y_t_x.forward(x_case=x_case, x_process=x_event_1, prefix_len=prefix_len)
        
        # t_reshaped = t.view(-1, 1)
        # y_ = y0 * (1 - t_reshaped) + y1 * t_reshaped
        
        # loss_y = self.outcome_distribution.loss(y, y_)
        # return loss_y
    
    # def get_loss(self, x_case, x_event, t, prefix_len, y):
    #     t_ = self.model_t_x.forward(x_case, x_event, prefix_len=prefix_len)
    #     if self.MODEL_PARAMS["ignore_x"]:
    #         x_case = torch.zeros_like(x_case)
    #         x_event = torch.zeros_like(x_event)

    #     # Make sure t1 has a 1 at the prefix_len position, zeros elsewhere (before prefix_len and after prefix_len), and make it the same sequence length as x_event
    #     t1 = torch.zeros(size=(x_event.shape[0], self.dims["dim_t"], x_event.shape[2]))
    #     t1[:, :, prefix_len.long() - 1] = 1

    #     # Make sure t0 has a 0 at the prefix_len position, zeros elsewhere (before prefix_len and after prefix_len)
    #     t0 = torch.zeros(size=(x_event.shape[0], self.dims["dim_t"], x_event.shape[2]))

    #     # Concatenate t to x_event
    #     x_event_1 = torch.cat((x_event, t1), dim=1)
    #     x_event_0 = torch.cat((x_event, t0), dim=1)

    #     y0 = self.model_y_t_x.forward(x_case, x_event_0, prefix_len=prefix_len)
    #     y1 = self.model_y_t_x.forward(x_case, x_event_1, prefix_len=prefix_len)

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