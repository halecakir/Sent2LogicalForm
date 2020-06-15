# Overview


## Data
[Geoquery data](https://github.com/halecakir/ThesisExperiments/tree/master/data) has been used. There are total 880 sentences-logical form pair (600 train, 280 test).

## Models

### Sequence-to-Sequence
![enter image description here](https://github.com/halecakir/Sent2LogicalForm/blob/master/extra/lstm.jpg)
###  Sequence-to-Sequence  + Attention

![enter image description here](https://github.com/halecakir/Sent2LogicalForm/blob/master/extra/attention.jpg)
## Attention Weight Matrix



### Sentence 1 : How high is m0?

![enter image description here](https://github.com/halecakir/Sent2LogicalForm/blob/master/extra/sample2.png)

### Sentence 2 : Which state boder s0?

![enter image description here](https://github.com/halecakir/Sent2LogicalForm/blob/master/extra/sample1.png)

### Base Implementation :
https://github.com/Alex-Fabbri/lang2logic-PyTorch
### Original Paper:
https://arxiv.org/pdf/1601.01280.pdf
