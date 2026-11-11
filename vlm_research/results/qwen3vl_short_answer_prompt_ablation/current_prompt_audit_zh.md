# 当前真实Prompt审计

本报告读取当前运行树的冻结 inventory、实际渲染函数及两卡原始 Checkpoint，未用历史配方代替。

GQA有明确短答指令：`Answer the question using a single word or phrase.`
ScienceQA-IMG、MMBench均以 `Please select the correct answer from the options above. ` 结尾；没有只答字母、禁止解释或“不重复选项”的指令。
ScienceQA/MMBench用户文本：Hint（如有）→Question→按原顺序Options→上述选择指令。原始Chat Template无额外短答System消息。
生成：Greedy，EOS=[151645,151643]；GQA cap32，ScienceQA/MMBench cap128。模型BF16/Eager、原始视觉预算不变。
15个例子的完整Prompt、Token IDs、输入Tensor哈希及Native/SVH逐字节一致性见 prompt_examples.jsonl。

历史Physical严格数值门禁仍为FAIL_RETAINED；本轮是独立Prompt协议，不改判历史失败。

## GQA / shortest / GQA:20705853
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What makes up the desk?
Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant

```
Token数=122; Token-ID SHA256=1595bf6b3dbfbb12cd77378ba8410b0a00e6733f0f215e1806d5eb565a105c20; Native/SVH输入完全一致。

## GQA / p25 / GQA:201482069
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Who is the red jacket covering?
Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant

```
Token数=287; Token-ID SHA256=3ef3d1ce86ca04713d60a0ad030581ed1f806d9f6d38d1d8a86e16613074cfd3; Native/SVH输入完全一致。

## GQA / median / GQA:201623452
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What is the appliance to the right of the appliance on the floor called?
Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant

```
Token数=295; Token-ID SHA256=c8fdbea4b2e4b331b417aa7d1147b95f97c005db8a51469caba7cac5ea4ed979; Native/SVH输入完全一致。

## GQA / p95 / GQA:201599698
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>What animal is it?
Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant

```
Token数=365; Token-ID SHA256=35da5126c29368bd93ee2fc86dbf9ee019be3f29ccecffc230a60220b84e9e36; Native/SVH输入完全一致。

## GQA / longest / GQA:202036678
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Is the green spinach to the left or to the right of the food the sausage is to the left of?
Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant

```
Token数=442; Token-ID SHA256=a4cd70b54d367fe3ec3d3a0ed27a51c19294889a24a8a1ee150bfca3fa8ed27c; Native/SVH输入完全一致。

## ScienceQA-IMG / shortest / ScienceQA-IMG:10391
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: Which material is this pool toy made of?
Options:
A. plastic
B. wool
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=106; Token-ID SHA256=da2585081cbf97fff424328ba71f3078f3179566dfccbdcc4731c11efedbb66c; Native/SVH输入完全一致。

## ScienceQA-IMG / p25 / ScienceQA-IMG:7759
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Hint: Figure: Steigerwald Forest.
The Steigerwald Forest is a temperate deciduous forest ecosystem in Bavaria, a state in southern Germany. This forest has many oak and beech trees.
Question: Which statement describes the Steigerwald Forest ecosystem?
Options:
A. It has soil that is poor in nutrients.
B. It has soil that is rich in nutrients.
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=169; Token-ID SHA256=dfda8911ee2587eb0513ced74adeb9196493d99875ab108458ab944b336f8ea8; Native/SVH输入完全一致。

## ScienceQA-IMG / median / ScienceQA-IMG:20998
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Hint: The passage below describes an experiment. Read the passage and then follow the instructions below.

Sasha cut tomato and broccoli plant leaves into one-inch squares. In each of 12 containers, she placed six leaf squares: three tomato-leaf squares and three broccoli-leaf squares. She put one slug from her garden into each container. After two days, Sasha measured the amount of each leaf square that had been eaten by the slugs. She compared the amount that had been eaten from the tomato-leaf squares to the amount that had been eaten from the broccoli-leaf squares.
Figure: a slug on a leaf.
Question: Identify the question that Sasha's experiment can best answer.
Options:
A. Do slugs weigh more after eating tomato leaves or broccoli leaves?
B. Do slugs eat more from tomato leaves or broccoli leaves?
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=271; Token-ID SHA256=c7a5631dd294333cd1e9227915f76327e07c9fb35aca3a0b1212abbb49aeb1f2; Native/SVH输入完全一致。

## ScienceQA-IMG / p95 / ScienceQA-IMG:6051
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: Which country is highlighted?
Options:
A. Saint Vincent and the Grenadines
B. The Bahamas
C. Cuba
D. Haiti
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=490; Token-ID SHA256=bb7c065e4be0c0adea41be916e21f218761c328c30e6b7e945ff0eb681c51e8c; Native/SVH输入完全一致。

## ScienceQA-IMG / longest / ScienceQA-IMG:3629
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Hint: Before sunrise on November 14, 1963, the crew of the fishing boat Isleifur II had just finished putting their fishing lines in the ocean off the southern coast of Iceland. As the crew waited to have breakfast, a strong smell of sulfur drifted over the boat. At first, crew members thought that the cook had burned the eggs or that something was wrong with the boat's engine. But when the sun started to rise, the crew saw black smoke billowing from the water a few kilometers away.
The captain of the Isleifur II first thought the smoke was coming from a boat that was on fire, so he sailed closer to try to help. But as the Isleifur II approached the column of smoke, the captain and crew saw glowing pieces of molten rock shooting up out of the water. The captain realized this was not a burning boat. It was a volcano erupting under the water!
Figure: the erupting undersea volcano seen by the sailors on the Isleifur II.
Question: According to the text, what evidence of a volcanic eruption did the captain observe?
Options:
A. He heard a report on the radio warning about a volcanic eruption.
B. He saw a nearby boat that was on fire.
C. He smelled sulfur and saw pieces of molten rock shooting out of the water.
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=561; Token-ID SHA256=20033396be7c801411f60680fef31239c2214f8dc39f95f21c7af466ecf3a938; Native/SVH输入完全一致。

## MMBench / shortest / MMBench:1001321
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: what emotion does this emoji express?
Options:
A. angry
B. happy
C. sad
D. excited
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=112; Token-ID SHA256=05b9d4ef8ff841f8e961577c07cc9e821e0a3970b2e8427e4d59d227c10e74e2; Native/SVH输入完全一致。

## MMBench / p25 / MMBench:2000292
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Hint: The images below show two pairs of magnets. The magnets in different pairs do not affect each other. All the magnets shown are made of the same material, but some of them are different shapes.
Question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?
Options:
A. The magnitude of the magnetic force is smaller in Pair 1.
B. The magnitude of the magnetic force is the same in both pairs.
C. The magnitude of the magnetic force is smaller in Pair 2.
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=195; Token-ID SHA256=73967da38ce663b09c9017847d51f4e7d569f8cbe05687932a796a9a0fdb3098; Native/SVH输入完全一致。

## MMBench / median / MMBench:3001143
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: What can be the relationship between the two persons in this image?
Options:
A. Mother and son
B. Brother and sister
C. Husband and wife
D. Father and daughter
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=238; Token-ID SHA256=2240cef35b8f88a902edcf514a39238c41756b155b3279abba03dfd07a2204f7; Native/SVH输入完全一致。

## MMBench / p95 / MMBench:1001924
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: What kind of human behavior does this picture describe?
Options:
A. A family is camping in the wilderness, pitching tents, building a fire, and enjoying the serenity of nature.
B. A man is surfing on a beach, riding the waves and feeling the rush of adrenaline while connecting with the power of the ocean.
C. A father in a suit takes his son overnight in the toilet, the son is already asleep while the father is secretly crying
D. A family is playing miniature golf on a course, putting their way through whimsical obstacles while laughing and having fun.
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=332; Token-ID SHA256=2c0a55b7829343993d5da83f26c87f74bbc5aac45c7e36721decd25688c1838c; Native/SVH输入完全一致。

## MMBench / longest / MMBench:3001651
```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: what code would generate this webpage in the browser?
Options:
A. <!DOCTYPE html>
<html>
<body>

<h2>JavaScript Objects</h2>

<p>There are two different ways to access an object property.</p>

<p>You can use person.property or person["property"].</p>

<p id="demo"></p>

<script>
// Create an object:
const person = {
firstName: "John",
lastName : "Doe",
id : 5566
};

// Display some data from the object:
document.getElementById("demo").innerHTML =
person["firstName"] + " " + person["lastName"];
</script>

</body>
</html>
B. <!DOCTYPE html>
<html>
<body>

<h2>JavaScript Objects</h2>
<p>An object method is a function definition, stored as a property value.</p>

<p id="demo"></p>

<script>
// Create an object:
const person = {
firstName: "John",
lastName: "Doe",
id: 5566,
fullName: function() {
return this.firstName + " " + this.lastName;
}
};

// Display data from the object:
document.getElementById("demo").innerHTML = person.fullName();
</script>

</body>
</html>
C. <!DOCTYPE html>
<html>
<body>

<h2>JavaScript Objects</h2>

<p>If you access an object method without (), it will return the function definition:</p>

<p id="demo"></p>

<script>
// Create an object:
const person = {
firstName: "John",
lastName : "Doe",
id : 5566,
fullName : function() {
return this.firstName + " " + this.lastName;
}
};

// Display data from the object:
document.getElementById("demo").innerHTML = person.fullName;
</script>

</body>
</html>
D. <!DOCTYPE html>
<html>
<body>

<h2>JavaScript Objects</h2>

<p id="demo"></p>

<script>
// Create an object:
const person = {
firstName: "John",
lastName: "Doe",
age: 50,
eyeColor: "blue"
};

// Display some data from the object:
document.getElementById("demo").innerHTML =
person.firstName + " is " + person.age + " years old.";
</script>

</body>
</html>
Please select the correct answer from the options above. 
<|im_end|>
<|im_start|>assistant

```
Token数=573; Token-ID SHA256=e9d8fcbdc33b3c2932eaa018bcf500e48b4b34c4b93ec1dd05fb21d235770bd4; Native/SVH输入完全一致。
