# speech_recognition_translation
## 1. Introduction
There are two function module in this project. One for speech recognition, which adopts Google WaveNet DNN, the other for 
machine translation, which adopts seq2seq + attention DNN. 

## 2. Environment
- python2.7
- tensorflow 1.0.0
- tflearn 0.3.2
- numpy 1.13.3
- six 1.11.0

## 3. Usage
Download the VCTK corpus and uncompress it to ./data/asr/. This corpus is used for training speech recognition model. 
Since it is an English corpus, the model will be able to recognise English speech. To train the speech recognition model:
```
python train_asr.py
```

To recognise a speech:
```
python recognise.py
```

The data for training machine translator has been put in ./data/nmt/. They are some subtitles of TED speech. Since it is 
an English-French corpus, the model can only translate English to French. To train the neural translation model:
```
python train_nmt.py
```

To translate a text:
```python
python translate.py
```

## 4. TODO
- do fine tuning for the model
- add BLEU score to check the model performance
- add beam search to NMT

## 5. Reference
- [How_to_make_a_language_translator](https://github.com/llSourcell/How_to_make_a_language_translator)
- [Speech-to-Text-WaveNet](https://github.com/buriburisuri/speech-to-text-wavenet)
- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
- [tensorflow-wavenet](https://github.com/Deeperjia/tensorflow-wavenet)

