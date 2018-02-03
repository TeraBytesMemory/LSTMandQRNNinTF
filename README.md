# LSTM and QRNN implemented in Tensorflow

抽象クラス定義ファイルの `abstract_neural_network.py` に従って書かれたLSTMとQRNNのコード

文字や単語をone-hotベクトル化する `text_vectorize.py` 付き

# Example usage

```python
import codecs

from abstract_neural_network import AbstractNeuralNetwork
from qrnn import QRNN
# or
# from lstm import LSTM
from text_vectorize import TextVectorize

HIDDEN_UNIT = 256
MAXLEN = 16
BATCH_SIZE = 64

with codecs.open('some_raw_text_file', 'r', 'utf-8') as f:
    text = f.read()

tv = TextVectorize(text)

model = QRNN(len(tv.char_to_id), len(tv.id_to_char), HIDDEN_UNIT, maxlen=MAXLEN)
# or
# model = LSTM(len(tv.char_to_id), len(tv.id_to_char), HIDDEN_UNIT, maxlen=MAXLEN)

itr = ((epoch, i)
       for epoch in range(100)
       for i in range(0, len(data), BATCH_SIZE))

for epoch, i in itr:
    model.fit(X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE], n_epoch=1, batch_size=BATCH_SIZE)

model.save("model.ckpt")
```
