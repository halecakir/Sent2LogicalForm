# Overview


## Data
[Geoquery data](https://github.com/halecakir/ThesisExperiments/tree/master/data) has been used. There are total 880 sentences-logical form pair (600 train, 280 test).

## Models

### Sequence-to-Sequence
![enter image description here](https://raw.githubusercontent.com/halecakir/ThesisExperiments/master/extra/lstm.jpg?token=ABFN2TYLIBAPOGRRTVSYAQ266BZ5U)
###  Sequence-to-Sequence  + Attention

![enter image description here](https://raw.githubusercontent.com/halecakir/ThesisExperiments/master/extra/attention.jpg?token=ABFN2T7TC4DJI2SEXGIG34K66BZ2M)
## Attention Weight Matrix



### Sentence 1 : How high is m0?

![enter image description here](https://raw.githubusercontent.com/halecakir/ThesisExperiments/master/extra/sample2.png?token=ABFN2TYLODJIBTRVWBR3MFK66BZ7Y)

### Sentence 2 : Which state boder s0?

![enter image description here](https://raw.githubusercontent.com/halecakir/ThesisExperiments/master/extra/sample1.png?token=ABFN2T5EFCM4H2K2IHGF5GK66B2AW)

Base Implementation :
https://github.com/Alex-Fabbri/lang2logic-PyTorch
Original Paper:
https://arxiv.org/pdf/1601.01280.pdf
