# -*- coding: utf-8 -*-
# @Time : 2021/10/11 16:48
# @Author : liujiachang
# @Email : liujiachang@thinkenergy.net.cn
# @File : tasks.py
import numpy as np
import torch

from utils import to_var


class Label:
    def __init__(self, column_name, training_set, sample_length=50):
        self.label = column_name
        self.sample_length = sample_length
        self.sample_mileage = [training_set[i][1][self.label] for i in range(self.sample_length)]
        self.max_mileage = max(self.sample_mileage)
        self.min_mileage = min(self.sample_mileage)

    def loss(self, batch, mean_pred, is_mse=True):
        label_data = []
        for i in batch[1][self.label]:
            norm_label = (i - self.min_mileage) / (self.max_mileage - self.min_mileage)
            label_data.append(norm_label)
        label = torch.tensor(label_data)
        x = mean_pred.squeeze().to("cuda")
        y = label.float().to("cuda")
        mse = torch.nn.MSELoss(reduction='mean')
        loss = 0
        if is_mse:
            loss = mse(x, y)
        return loss


class Task:
    def __init__(self, columns, encoder_dimension=122, decoder_dimension=122,
                 output_dimension=122, task_name='ev'):
        self.encoder_dimension = encoder_dimension
        self.decoder_dimension = decoder_dimension
        self.output_dimension = output_dimension
        self.task_name = task_name
        self.columns = columns
        self.encoder = []
        self.decoder = []
        self.target = []
        eval(self.task_name.capitalize() + 'Task.set_params')(self)
        eval(self.task_name.capitalize() + 'Task.get_task_idx')(self, columns)

    def encoder_filter(self, input_embedding):
        # Extracting input dimensions
        return eval(self.task_name.capitalize() + 'Task.task_encoder')(self, input_embedding, self.columns)

    def decoder_filter(self, input_embedding):
        # Escape device
        self.decoder = self.encoder[:self.decoder_dimension]
        return to_tensor(to_array(input_embedding)[:, :, self.decoder])

    def target_filter(self, input_embedding):
        # Output target dimension
        self.target = self.encoder[self.decoder_dimension:]
        return to_tensor(to_array(input_embedding)[:, :, self.target])

    def task_encoder(self, input_embedding, columns):
        return to_tensor(to_array(input_embedding)[:, :, self.encoder])


class EvTask(Task):
    def set_params(self):
        """
        The number of dimensions
        """
        self.encoder_dimension = 6
        self.decoder_dimension = 2
        self.output_dimension = 4

    def get_task_idx(self, columns):
        """
        Extract the specified column
        Args:
            columns
        Return:
            Specifies the corresponding subscript for the column
        """
        self.encoder = np.array(
            [columns.index("soc"), columns.index("current"),
             columns.index("max_temp"), columns.index("max_single_volt"),
             columns.index("min_single_volt"), columns.index("volt")]).astype(int)
        return self.encoder

def to_tensor(input_embedding):
    return to_var(torch.from_numpy(np.array(input_embedding)))


def to_array(input_embedding):
    return input_embedding.cpu().numpy()
