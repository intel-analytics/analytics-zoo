## **Maximum**
Layer that computes the maximum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

**Scala:**
```scala
Maximum()
```
**Python:**
```python
Maximum()
```

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Maximum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = Maximum[Float]().inputs(Array(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.0085953	1.1095089	1.7487661	1.576811	1.3192933	1.173145	1.7567515	1.750411	
	   1.0303572	1.0285444	1.4724362	1.0070276	1.6837391	1.2812499	1.7207997	1.9301186	
	   1.6642286	1.300531	1.2989123	1.0117699	1.5870146	1.2845709	1.9443712	1.1186409	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.6898564	0.98180896	0.22475463	0.44690642	0.22631128	0.8658154	0.96297216	0.038640756	
	   0.33791444	0.35920507	0.2056811	0.97009206	0.891668	0.73843783	0.49456882	0.92106706	
	   0.54771185	0.52310455	0.49114317	0.93534994	0.82244986	0.080847055	0.56450963	0.73846775	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
1.0085953	1.1095089	1.7487661	1.576811	1.3192933	1.173145	1.7567515	1.750411	
1.0303572	1.0285444	1.4724362	1.0070276	1.6837391	1.2812499	1.7207997	1.9301186	
1.6642286	1.300531	1.2989123	1.0117699	1.5870146	1.2845709	1.9443712	1.1186409	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import Maximum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = Maximum()([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.85046637 0.76454759 0.92092265 0.18948392 0.96141892 0.75558563
  0.16956892 0.49839472]
 [0.36737777 0.25567011 0.36751645 0.49982497 0.62344662 0.10207675
  0.14432582 0.09316922]
 [0.34775348 0.56521665 0.01922694 0.97405856 0.96318355 0.48008106
  0.09525403 0.64539933]]

input2:
[[0.23219699 4.58298671 4.08334902 3.35729794 3.28995515 3.88572392
  0.13552906 2.20767025]
 [4.41043478 0.74315223 1.57928439 4.06317265 4.35646267 4.43969778
  0.64163024 0.14681471]
 [1.60829488 3.75488617 4.69265858 1.38504037 3.2210222  3.4321568
  4.00735856 2.6106414 ]]
```
Output is
```python
[[0.8504664  4.582987   4.083349   3.357298   3.2899551  3.8857238
  0.16956893 2.2076702 ]
 [4.4104347  0.74315226 1.5792844  4.063173   4.3564625  4.4396977
  0.64163023 0.1468147 ]
 [1.6082948  3.7548862  4.6926584  1.3850404  3.2210221  3.4321568
  4.0073586  2.6106415 ]]
```

---
## **maximum**
Functional interface to the `Maximum` layer.

**Scala:**
```scala
maximum(inputs)
```
**Python:**
```python
maximum(inputs)
```

**Parameters:**

* `inputs`: A list of input tensors (at least 2).
* `**kwargs`: Standard layer keyword arguments.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Maximum.maximum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = maximum(inputs = List(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.5386189	1.67534	1.3651735	1.0366004	1.2869223	1.6384993	1.5557045	1.5723307	
	   1.2382979	1.0155076	1.1055984	1.1010389	1.6874355	1.3107576	1.2041453	1.9931196	
	   1.4011493	1.0774659	1.3888124	1.7762307	1.8265619	1.7934192	1.7732148	1.2978737	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.14446749	0.7428541	0.9886685	0.5107685	0.85201174	0.40988243	0.12447342	0.8556565	
	   0.91737056	0.35073906	0.07863916	0.89909834	0.8177192	0.09691833	0.1997524	0.4406145	
	   0.4190805	0.6956053	0.9765333	0.6748145	0.87814146	0.5421859	0.31012502	0.25200275	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
1.5386189	1.67534	1.3651735	1.0366004	1.2869223	1.6384993	1.5557045	1.5723307	
1.2382979	1.0155076	1.1055984	1.1010389	1.6874355	1.3107576	1.2041453	1.9931196	
1.4011493	1.0774659	1.3888124	1.7762307	1.8265619	1.7934192	1.7732148	1.2978737	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import maximum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = maximum([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.32837152 0.66842081 0.5893283  0.71063029 0.53254716 0.98882168
  0.53400631 0.93659819]
 [0.6198554  0.51117444 0.74729989 0.65475831 0.70510429 0.87443468
  0.5629698  0.285089  ]
 [0.43159809 0.84360242 0.8493521  0.78723246 0.35496674 0.00144353
  0.07231955 0.76153367]]
  
 input2: 
[[4.00763759 0.37730923 3.88563172 2.22099527 3.38980926 2.84321074
  0.29846632 4.07808143]
 [0.36804983 2.34995472 2.24190514 1.63816757 2.22642342 1.45099988
  0.55931613 0.42101343]
 [0.30218586 2.75409562 0.24024987 3.89805855 4.57479762 2.6592906
  2.38562566 1.46560388]]
```
Output is
```python
[[4.0076375  0.6684208  3.8856318  2.2209952  3.3898094  2.8432107
  0.5340063  4.0780816 ]
 [0.6198554  2.3499546  2.2419052  1.6381676  2.2264235  1.4509999
  0.5629698  0.42101344]
 [0.4315981  2.7540956  0.8493521  3.8980587  4.5747976  2.6592906
  2.3856256  1.4656038 ]]
```

---
## **Minimum**
Layer that computes the minimum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

**Scala:**
```scala
Minimum()
```
**Python:**
```python
Minimum()
```

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Minimum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = Minimum[Float]().inputs(Array(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.9953886	1.0161483	1.844671	1.1757553	1.7548938	1.4735664	1.981268	1.354598	
	   1.786057	1.4920603	1.538079	1.6601591	1.5213481	1.9032607	1.5938802	1.9769413	
	   1.428338	1.5083437	1.1141979	1.4320385	1.9785057	1.845624	1.0637122	1.8684102	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.6624951	0.71156764	0.004659928	0.8797748	0.8676378	0.5605965	0.03135305	0.3550916	
	   0.86810714	0.26216865	0.8639284	0.3357767	0.22505952	0.8216017	0.74407136	0.73391193	
	   0.74810994	0.11495259	0.89162785	0.93693215	0.5673804	0.20798753	0.022446347	0.36790285	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
0.6624951	0.71156764	0.004659928	0.8797748	0.8676378	0.5605965	0.03135305	0.3550916	
0.86810714	0.26216865	0.8639284	0.3357767	0.22505952	0.8216017	0.74407136	0.73391193	
0.74810994	0.11495259	0.89162785	0.93693215	0.5673804	0.20798753	0.022446347	0.36790285	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import Minimum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = Minimum()([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.15979525 0.48601263 0.10587506 0.61769843 0.26736246 0.64769634
  0.01616307 0.93659085]
 [0.3412241  0.02449786 0.64638927 0.32875475 0.77737532 0.94151168
  0.95571165 0.1285685 ]
 [0.9758039  0.89746475 0.84606271 0.87471803 0.80568297 0.85872464
  0.77484317 0.73048055]]

input2:
[[0.99780609 1.48670819 0.08911578 2.68460415 1.21065202 1.82819649
  2.91991375 1.07241835]
 [3.18491884 3.72856744 3.82128444 1.53010301 1.20795887 3.20653343
  3.07794378 1.59084261]
 [4.39776482 3.37465746 0.23752302 3.47325532 2.38110537 4.64806043
  3.99013359 0.56055062]]
```
Output is
```python
[[0.15979525 0.48601264 0.08911578 0.61769843 0.26736248 0.6476963
  0.01616307 0.93659085]
 [0.3412241  0.02449786 0.64638925 0.32875475 0.77737534 0.9415117
  0.95571166 0.1285685 ]
 [0.9758039  0.89746475 0.23752302 0.874718   0.80568296 0.85872465
  0.77484316 0.56055063]]
```

---
## **minimum**
Functional interface to the `Minimum` layer.

**Scala:**
```scala
minimum(inputs)
```
**Python:**
```python
minimum(inputs)
```

**Parameters:**

* `inputs`: A list of input tensors (at least 2).
* `**kwargs`: Standard layer keyword arguments.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Minimum.minimum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

val input1 = Tensor[Float](3, 8).rand(0, 1)
val input2 = Tensor[Float](3, 8).rand(1, 2)
val input = T(1 -> input1, 2 -> input2)
val l1 = Input[Float](inputShape = Shape(8))
val l2 = Input[Float](inputShape = Shape(8))
val layer = minimum(inputs = List(l1, l2))
val model = Model[Float](Array(l1, l2), layer)
val output = model.forward(input)
```
Input is:
```scala
input: {
	2: 1.0131017	1.7637167	1.3681185	1.6208028	1.2059574	1.967363	1.5065156	1.5110291	
	   1.1055611	1.4148856	1.5531528	1.3481603	1.3744175	1.5192658	1.7290237	1.629003	
	   1.5601189	1.4540797	1.0981613	1.2463317	1.9510872	1.0527081	1.0487831	1.4148198	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
	1: 0.96061057	0.41263154	0.5029265	0.28855637	0.8030459	0.5923882	0.93190056	0.15111573	
	   0.54223496	0.37586558	0.63049513	0.32910138	0.029513072	0.017590795	0.1943584	0.77225924	
	   0.21727595	0.6552713	0.899118	0.07937545	0.016797619	0.5491529	0.7383374	0.8877089	
	   [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
 }
```
Output is:
```scala
output: 
0.96061057	0.41263154	0.5029265	0.28855637	0.8030459	0.5923882	0.93190056	0.15111573	
0.54223496	0.37586558	0.63049513	0.32910138	0.029513072	0.017590795	0.1943584	0.77225924	
0.21727595	0.6552713	0.899118	0.07937545	0.016797619	0.5491529	0.7383374	0.8877089	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import minimum
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[8])
l2 =Input(shape=[8])
layer = minimum([l1, l2])
input1 = np.random.random([3, 8])
input2 = 5 * np.random.random([3, 8])
model = Model([l1, l2], layer)
output = model.forward([input1, input2])
```
Input is:
```python
input1:
[[0.82114595 0.19940446 0.57764964 0.49092517 0.67671559 0.72676631
  0.85955214 0.80265871]
 [0.05640074 0.17010821 0.4896911  0.14905843 0.33233282 0.82684842
  0.58635163 0.10010479]
 [0.83053659 0.83788089 0.6177536  0.71670009 0.54454425 0.19431431
  0.49180683 0.25640596]]

input2:
[[0.47446558 3.8752243  4.9299194  3.35971335 0.85980843 2.37388383
  4.38802943 4.3253041 ]
 [2.65459389 2.93173369 3.6176582  0.75475853 0.62484204 4.16820336
  3.24864692 1.42238813]
 [0.439386   2.43623362 0.20248675 1.60213208 1.08081789 0.59718494
  0.29896311 0.73010527]]
```
Output is
```python
[[0.47446558 0.19940446 0.57764965 0.49092516 0.6767156  0.7267663
  0.85955215 0.80265874]
 [0.05640074 0.17010821 0.4896911  0.14905843 0.33233282 0.82684845
  0.58635163 0.10010479]
 [0.439386   0.8378809  0.20248675 0.7167001  0.5445443  0.1943143
  0.2989631  0.25640595]]
```

## **Average**
Layer that computes the average (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

**Scala:**
```scala
Average()
```
**Python:**
```python
Average()
```

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Input, InputLayer, Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Average.average

val input1 = Tensor[Float](3, 10).rand(0, 1)
val input2 = Tensor[Float](3, 10).rand(1, 2)
val input3 = Tensor[Float](3, 10).rand(2, 3)
val input = T(1 -> input1, 2 -> input2, 3 -> input3)
val l1 = Input[Float](inputShape = Shape(10))
val l2 = Input[Float](inputShape = Shape(10))
val l3 = Input[Float](inputShape = Shape(10))
val layer = Average[Float]().inputs(Array(l1, l2, l3))
val model = Model[Float](Array(l1, l2, l3), layer)
model.getOutputShape().toSingle().toArray should be (Array(-1, 10))
model.forward(input) should be ((input1 + input2 + input3)/3)
```
Input is:
```scala
input: {
3: 2.1388125	2.6964622	2.8276837	2.3661323	2.4378736	2.2604358	2.465229	2.9974892	2.3736873	2.9525855	
       2.9544477	2.0917578	2.5748422	2.8470073	2.8541746	2.8388956	2.4446492	2.3847318	2.772636	2.9858146	
       2.923554	2.4356844	2.1714668	2.496371	2.6633065	2.074316	2.1330173	2.702345	2.1016605	2.4054792	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
	2: 1.7939124	1.8057413	1.991265	1.2754726	1.8769448	1.3218789	1.4537191	1.144134	1.516381	1.1731085	
       1.1079352	1.0502894	1.6418922	1.3537997	1.4794713	1.1901937	1.4494545	1.0795411	1.661124	1.0139834	
       1.5142206	1.4812648	1.4023355	1.0209563	1.312892	1.0843754	1.1938657	1.16379	1.9043504	1.2607019	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
	1: 0.44098258	0.76005894	0.22524318	0.1684992	0.0051373183	0.21741198	0.2731409	0.35371205	0.05021227	0.884149	
       0.84959924	0.8669313	0.029727593	0.35729104	0.6929461	0.41323605	0.37593237	0.73493475	0.84717196	0.89586157	
       0.11885294	0.15163219	0.5033071	0.7372261	0.41698205	0.87994254	0.6085509	0.3116428	0.1701491	0.78339463	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
}
```
Output is:
```scala
output: 
1.4579027	1.7540874	1.6813973	1.2700348	1.4399853	1.2665756	1.3973631	1.4984453	1.3134269	1.6699476	
1.6373274	1.3363261	1.4154873	1.519366	1.6755308	1.4807751	1.4233454	1.3997358	1.7603106	1.6318865	
1.5188758	1.3561939	1.3590364	1.4181845	1.4643935	1.3462113	1.3118113	1.3925927	1.3920534	1.4831918	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]

```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import Average
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[10])
l2 =Input(shape=[10])
l3 =Input(shape=[10])
layer = Average()([l1, l2, l3])
input1 = np.random.random([3, 10])
input2 = 5 * np.random.random([3, 10])
input3 = np.random.random([3, 10])
model = Model([l1, l2, l3], layer)
output = model.forward([input1, input2, input3])
```
Input is:
```python
input1:
[[0.41363052 0.6743329  0.145657   0.6637766  0.66581595 0.6749803 
  0.44645357 0.20913158	0.37975723 0.9108898]	
 [0.8512118	 0.11105641	0.8040392  0.52162653 0.64165527 0.7338619
  0.87358207 0.45065743	0.3629784  0.98838156]	
 [0.98552185 0.44495004	0.45191813 0.63229626 0.8232937	 0.68438464
  0.45459357 1.2166076E-4 0.838354	0.85861313]]	
input2:
[[1.2572492  1.7911074  1.552091   1.3164328  1.5672878	1.2264123
  1.0933136	 1.1894734	1.7110546  1.4871693]	
 [1.2493236	 1.0981448	1.2093825  1.385048	  1.7991946	1.7278074
  1.3594079	 1.7045076	1.010181   1.8572828]	
 [1.9143413	 1.2307098	1.4673012  1.0212818  1.1806543	1.3802769
  1.8570768	 1.117075	1.5691379  1.9965043]]	
input3:
[[2.8687525	 2.257423	2.5164936  2.2316918  2.444299	2.791102
  2.3251874	 2.64277	2.094336   2.0139678]	
 [2.80216	 2.7153122	2.693267   2.8926282  2.760983	2.4018617
  2.3492196	 2.5027466	2.3177707  2.1410158]	
 [2.8840475	 2.4416642	2.5429523  2.5207841  2.7137184	2.0619774
  2.0841057	 2.2143555	2.4883647  2.284767]]	

```
Output is
```python
[[1.5132108	 1.5742878	1.4047472  1.4039671  1.5591342	 1.5641649
  1.2883182	 1.347125	1.3950493  1.4706757]	
 [1.6342319	 1.3081712	1.5688963  1.5997677  1.7339443	 1.6211771
  1.5274032	 1.5526371	1.2303101  1.6622267]	
 [1.9279703	 1.3724413	1.4873905  1.3914541  1.5725555	 1.3755463
  1.4652586	 1.1105174	1.6319523  1.7132947]]	

```

---
## **average**
Functional interface to the `Average` layer.

**Scala:**
```scala
average(inputs)
```
**Python:**
```python
average(inputs)
```

**Parameters:**

* `inputs`: A list of input tensors (at least 2).
* `**kwargs`: Standard layer keyword arguments.

**Scala example:**
```scala

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Input, InputLayer, Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Average.average
val input1 = Tensor[Float](3, 10).rand(0, 1)
val input2 = Tensor[Float](3, 10).rand(1, 2)
val input3 = Tensor[Float](3, 10).rand(2, 3)
val input = T(1 -> input1, 2 -> input2, 3 -> input3)
val l1 = Input[Float](inputShape = Shape(10))
val l2 = Input[Float](inputShape = Shape(10))
val l3 = Input[Float](inputShape = Shape(10))
val layer = average(inputs = List(l1, l2, l3))
val model = Model[Float](Array(l1, l2, l3), layer)
model.getOutputShape().toSingle().toArray should be (Array(-1, 10))
model.forward(input) should be ((input1 + input2 + input3)/3)
```
Input is:
```scala
input: {
3: 2.8428721	2.2226758	2.5208285	2.2367554	2.4270968	2.5545428	2.1287117	2.6222005	2.431273	2.8214326	
       2.8859177	2.3255236	2.9590778	2.9784636	2.2240565	2.0665257	2.242283	2.740618	2.0167441	2.0605018	
       2.026233	2.6200292	2.2228913	2.8658607	2.199205	2.5279496	2.2850516	2.2320945	2.8136265	2.2246487	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
	2: 1.7467746	1.8021997	1.6509205	1.6072322	1.4914033	1.5677507	1.5366478	1.7563739	1.5117183	1.6645014	
       1.5314101	1.3897215	1.7508979	1.9116509	1.8281436	1.8060269	1.6515387	1.0517279	1.8487817	1.3851342	
       1.7522845	1.388323	1.2490842	1.746568	1.0954672	1.7134392	1.4903576	1.3686041	1.6109412	1.079587	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
	1: 0.9149919	0.20087978	0.5228629	0.2721904	0.671595	0.9914089	0.08755763	0.06214724	0.24063993	0.30998093	
       0.93342364	0.035474923	0.60141927	0.055402536	0.6627572	0.66029215	0.052742954	0.84818095	0.48247424	0.8988862	
       0.019309381	0.8240607	0.5062556	0.2224868	0.07100526	0.27008608	0.04632365	0.40351728	0.49034202	0.76545227	
       [com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
 }
```
Output is:
```scala
output: 
1.8348796	1.4085851	1.5648706	1.3720593	1.5300317	1.7045674	1.2509725	1.4802406	1.3945436	1.5986383	
1.7835839	1.2502401	1.7704649	1.6485057	1.5716524	1.5109482	1.3155216	1.5468423	1.4493334	1.4481742	
1.2659423	1.6108043	1.3260771	1.6116384	1.1218925	1.503825	1.273911	1.3347386	1.6383032	1.3565626	
[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x10]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Model
from zoo.pipeline.api.keras2.layers import average
from zoo.pipeline.api.keras.layers import Input

l1 = Input(shape=[10])
l2 =Input(shape=[10])
l3 =Input(shape=[10])
layer = average([l1, l2, l3])
input1 = np.random.random([3, 10])
input2 = 5 * np.random.random([3, 10])
model = Model([l1, l2, l3], layer)
output = model.forward([input1, input2, input3])
```
Input is:
```python
input1:
[[0.02466183 0.27168578	0.2550742  0.21883814 0.82554376 0.28807813
  0.13608417 0.09758039	0.5034134  0.15892369]	
 [0.87264824 0.14789267	0.9338583  0.57726556 0.8708862	 0.5165683
  0.17245324 0.9322028	0.41752812 0.12310909]	
 [0.79332834 0.5849009	0.99154574 0.3219758  0.2697678	 0.20010926
  0.26830813 0.2019596	0.6096836  0.03093836]]	
input2:
[[1.4111987  1.198054	1.842369   1.8801643  1.4168038	 1.3551488
  1.975179	 1.9757051	1.1127443  1.4476182]	
 [1.232829	 1.7522926	1.532449   1.3352888  1.2850991	 1.0257217
  1.7294677	 1.308971	1.2973334  1.6694721]	
 [1.8759854	 1.4279369	1.5563112  1.5380665  1.6536583	 1.2297603
  1.45884	 1.8191999	1.118961   1.1868691]]	
input3:
[[2.519016	 2.339481	2.4914405  2.0135844  2.8932412	 2.158044
  2.9289017	 2.0235717	2.2571347  2.2399058]	
 [2.1186874	 2.843763	2.7362454  2.2478065  2.3966339	 2.4745004
  2.9351826	 2.6583135	2.8862307  2.4307551]	
 [2.2603345	 2.7962108	2.2940488  2.6015089  2.7432284	 2.3210974
  2.3186648	 2.4515545	2.9862666  2.1030922]]	
```
Output is
```python
[[1.3182921	 1.2697403	1.5296278  1.3708624  1.7118629	 1.2670903
  1.680055	 1.3656191	1.2910975  1.2821493]	
 [1.4080551	 1.5813162	1.7341843  1.3867869  1.5175397	 1.3389301
  1.612368	 1.6331625	1.5336975  1.4077787]	
 [1.6432161	 1.6030163	1.6139686  1.4871838  1.5555515	 1.2503223
  1.3486044	 1.4909047	1.5716372  1.1069666]]	
```