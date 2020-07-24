# language_layers

A experimental tool for text encoding research.


### Lexical encoding

```

$python lexical_encoder.py 

Building SubwordTextEncoder...
Setting subword encoder...
Getting subword encoder...
['de_', ', ', 'a_', 'o_', 'la_', 's_', '. ', 'y_', 'que_', 'en_']
Setting subword probability and entropy distributions...
Saving LexicalEncoder instance...
Loading LexicalEncoder instance...
Loaded lexcoder <class '__main__.LexicalEncoder'>
text:
 este es el texto de prueba
Getting subword encoder...
LexEncoded text with padding:
 [0.0, 0.13404515619996385, ... ]
LexEncoded text without padding:
 [0.026436286750756564, 0.04983251327071346, 0.041314640497261586, 0.06262181136218017, 0.0224592277175346, 0.011863952758310583, 0.018908783594966498, 0.13404515619996385, 0.01429372442409675, 0.01963422926919046, 0.024967786553605355]

```

### Semantic encoding

```
$python word2vec_encoder.py 

Building vocabulary...
model.wv.vocab size 4232
Training model doc2vec model...
text:
 este es el texto de prueba
Semantic encoded text:
 [-0.01229797  0.01086341  0.01195752 -0.004056   -0.00546786 -0.02437639
  0.0055791   0.00323276 -0.01213035  0.00332347 -0.01610198  0.00580948
  0.00371256  0.00495905 -0.00113193  0.0140602  -0.00754718  0.02568262
 -0.00619837  0.01084981  0.00288903 -0.00914368 -0.00033934  0.00206474
  0.00826169 -0.01210871  0.01225642 -0.00059115 -0.00193947  0.00652603
 -0.00465229 -0.00696411 -0.00137275  0.02214912 -0.00101232 -0.0033177
 -0.02691323 -0.0064485   0.00978336 -0.01573619]
```

### Prerequisites

Verify you have a Python 3 installation or use a virtualenv.


```

$ python --version

Python 3.7.1

```

### Dependencies

See requirements.txt


```
tensorflow==2.2.0
tensorflow-datasets==3.2.1
gensim==3.8.3
Cython==0.29.21
pandas==1.0.5

```


### Data

Sample data is provided.

```
~/repo/language_layers$ls -R data
2019/

data/2019:
septiembre/

data/2019/septiembre:
vértigo.csv              yuc_la_jornada_maya.csv
```

## Authors

* **Alejandro Molina-Villegas** - [Conacyt-CentroGeo](http://mid.geoint.mx/site/integrante/id/15.html)

* **Edwin Aldana Bobadilla** - []()

* **Melesio Crespo Sanchez** - []()


## License

   Copyright 2019 Alejandro Molina-Villegas

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
