from zoo.zouwu.model.forecast.model.base_keras_model import KerasBaseModel
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, LSTMCell
import tensorflow as tf

def model_creator(config):
    # config
    input_feature_num = config["input_feature_num"]
    output_feature_num = config["output_feature_num"]
    future_seq_len = config["future_seq_len"]
    lstm_hidden_dim = config.get("lstm_hidden_dim", 128)
    lstm_layer_num = config.get("lstm_layer_num", 2)
    dropout_rate = config.get("dropout", 0.25)
    fc_layer_num = config.get("fc_layer_num", 2)
    fc_hidden_dim = config.get("fc_hidden_dim", 128)
    optim = config.get("optim", "Adam")
    lr = config.get("lr", 0.001)
    metrics = [config["metric"]]
    loss = config.get("loss", "mse")
    
    # model definition 
    model_input = Input(shape=(None, input_feature_num))
    for i in range(lstm_layer_num):
        return_sequences = True if i != lstm_layer_num - 1 else False
        return_state = not return_sequences
        lstm_input = model_input if i == 0 else dropout
        lstm = LSTM(units=lstm_hidden_dim, return_sequences=return_sequences, return_state=return_state)(lstm_input)
        dropout = Dropout(rate=dropout_rate)(lstm)
    decoder = LSTMCell(lstm_hidden_dim)
    decode_list = []
    h, c = dropout[1], dropout[2]
    for i in range(future_seq_len):
        dropout = decoder(h, [h, c])
        decode_list.append(dropout[0])
        h, c = dropout[0], dropout[1][1]
    decode = tf.stack(decode_list, axis=1)
    for i in range(fc_layer_num):
        outdim = fc_hidden_dim if i < fc_layer_num - 1 else output_feature_num
        decode = Dense(outdim)(decode)
    model = Model(model_input, decode)
    model.compile(loss=loss,
                  metrics=metrics,
                  optimizer=getattr(tf.keras.optimizers, optim)(lr=lr))
    return model

class LSTMSeq2Seq(KerasBaseModel):
    def __init__(self, check_optional_config=False, future_seq_len=1):
        super(LSTMSeq2Seq, self).__init__(model_creator=model_creator,
                                          check_optional_config=check_optional_config)

    def _check_config(self, **config):
        super()._check_config(**config)
        # TODO: add check
        pass
    
    def _get_required_parameters(self):
        return {"input_feature_num",
                "future_seq_len",
                "output_feature_num"
                } | super()._get_required_parameters()

    def _get_optional_parameters(self):
        return {"lstm_hidden_dim",
                "lstm_layer_num",
                "fc_layer_num",
                "fc_hidden_dim",
                "dropouts",
                "optim",
                "lr"
                } | super()._get_optional_parameters()

if __name__ == "__main__":
    config = {"lstm_layer_num": 2, 
              "lstm_hidden_dim": 128,
              "fc_layer_num": 2,
              "fc_hidden_dim": 128,
              "input_seq_len": 10,
              "input_feature_num": 8,
              "output_feature_num": 4,
              "future_seq_len": 5,
              "dropout": 0.1,
              "metric": "mse"}
    inputs = tf.random.normal([32, 10, 8])
    model = model_creator(config)
    output = model(inputs)
    print(output.shape)
